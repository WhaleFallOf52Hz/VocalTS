import logging
import multiprocessing
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Set this to the GPU IDs you want to use
import time
import glob
import re

warnings.filterwarnings(
    "ignore",
    message=".*The pynvml package is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()


def _latest_numeric_ckpt(model_dir, prefix):
    pattern = re.compile(rf"^{prefix}_(\d+)\.pth$")
    candidates = []
    for path in glob.glob(os.path.join(model_dir, f"{prefix}_[0-9]*.pth")):
        name = os.path.basename(path)
        match = pattern.match(name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None, None
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def _latest_last_ckpt_pair(model_dir):
    pattern = re.compile(r'^(G|D)_last_(\d+)\.pth$')
    grouped = {}
    for file_name in os.listdir(model_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        prefix, step_str = match.groups()
        step = int(step_str)
        if step not in grouped:
            grouped[step] = {}
        grouped[step][prefix] = os.path.join(model_dir, file_name)

    candidates = []
    for step, files in grouped.items():
        if 'G' in files and 'D' in files:
            candidates.append((step, files['G'], files['D']))

    if candidates:
        step, g_path, d_path = max(candidates, key=lambda item: item[0])
        return g_path, d_path, step

    g_legacy = os.path.join(model_dir, "G_last.pth")
    d_legacy = os.path.join(model_dir, "D_last.pth")
    if os.path.isfile(g_legacy) and os.path.isfile(d_legacy):
        return g_legacy, d_legacy, None

    return None, None, None


def _prune_last_ckpt_pairs(model_dir, keep_last=1):
    keep_last = max(1, int(keep_last))
    pattern = re.compile(r'^(G|D)_last_(\d+)\.pth$')
    grouped = {}
    for file_name in os.listdir(model_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        prefix, step_str = match.groups()
        step = int(step_str)
        if step not in grouped:
            grouped[step] = {}
        grouped[step][prefix] = file_name

    complete_steps = []
    for step, files in grouped.items():
        if 'G' in files and 'D' in files:
            complete_steps.append(step)

    complete_steps.sort(reverse=True)
    for step in complete_steps[keep_last:]:
        for prefix in ('G', 'D'):
            file_name = grouped[step][prefix]
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)


def _list_best_ckpt_pairs(model_dir):
    pattern = re.compile(r'^(G|D)_best_(\d+)_([0-9]+(?:\.[0-9]+)?)\.pth$')
    grouped = {}
    for file_name in os.listdir(model_dir):
        match = pattern.match(file_name)
        if not match:
            continue
        prefix, step_str, loss_str = match.groups()
        key = (int(step_str), float(loss_str))
        if key not in grouped:
            grouped[key] = set()
        grouped[key].add(prefix)

    pairs = []
    for (step, loss), prefixes in grouped.items():
        if 'G' in prefixes and 'D' in prefixes:
            pairs.append((step, loss))
    pairs.sort(key=lambda item: (item[1], item[0]))
    return pairs

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    logger = None
    writer = None
    writer_eval = None
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
    
    # for pytorch on win, backend use gloo    
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, split="train", all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, split="val", all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])

    skip_optimizer = False
    resume_success = False
    resume_error_messages = []

    g_last_path, d_last_path, last_step = _latest_last_ckpt_pair(hps.model_dir)
    g_numeric_path, numeric_step_g = _latest_numeric_ckpt(hps.model_dir, "G")
    d_numeric_path, numeric_step_d = _latest_numeric_ckpt(hps.model_dir, "D")

    resume_candidates = []
    if g_last_path is not None and d_last_path is not None:
        resume_candidates.append(("last", g_last_path, d_last_path, last_step))
    if g_numeric_path is not None and d_numeric_path is not None:
        numeric_step = min(numeric_step_g, numeric_step_d)
        resume_candidates.append(("numeric", g_numeric_path, d_numeric_path, numeric_step))

    for source_tag, g_resume_path, d_resume_path, resume_step in resume_candidates:
        try:
            _, _, _, epoch_str = utils.load_checkpoint(g_resume_path, net_g,
                                                       optim_g, skip_optimizer)
            _, _, _, epoch_str = utils.load_checkpoint(d_resume_path, net_d,
                                                       optim_d, skip_optimizer)
            epoch_str = max(epoch_str, 1)
            if resume_step is not None:
                global_step = resume_step + 1
            else:
                global_step = max(0, (epoch_str - 1) * len(train_loader))
            resume_success = True
            if rank == 0:
                logger.info(f"Resume[{source_tag}] from G={g_resume_path}, D={d_resume_path}, epoch={epoch_str}, global_step={global_step}")
            break
        except Exception as e:
            resume_error_messages.append(f"{source_tag}: {repr(e)}")

    if not resume_success:
        print("load old checkpoint failed...")
        if rank == 0 and len(resume_candidates) == 0:
            logger.warning(f"No valid resume candidates found in {hps.model_dir}; fallback to fresh training")
        if rank == 0 and len(resume_error_messages) > 0:
            for message in resume_error_messages:
                logger.warning(f"Resume failed - {message}")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    if rank == 0:
        writer = SummaryWriter(log_dir=hps.model_dir, purge_step=global_step)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"), purge_step=global_step)

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        # update learning rate
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    checkpoint_save_interval = int(getattr(hps.train, 'save_interval', 10000))
    best_ckpts_topk = int(getattr(hps.train, 'best_ckpts_topk', 3))
    checkpoint_save_interval = max(1, checkpoint_save_interval)
    best_ckpts_topk = max(1, best_ckpts_topk)

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv,volume = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        
        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths,vol = volume)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        

        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False, dtype=half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.module.use_automatic_f0_prediction else 0
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss=0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.module.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                eval_loss_g_total = evaluate(hps, net_g, net_d, eval_loader, writer_eval)
                last_step = global_step
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, f"G_last_{last_step}.pth"))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, f"D_last_{last_step}.pth"))
                _prune_last_ckpt_pairs(hps.model_dir, keep_last=1)

                best_g_path = os.path.join(hps.model_dir, f"G_best_{global_step}_{eval_loss_g_total:.6f}.pth")
                best_d_path = os.path.join(hps.model_dir, f"D_best_{global_step}_{eval_loss_g_total:.6f}.pth")
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, best_g_path)
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, best_d_path)
                utils.prune_best_checkpoints(path_to_models=hps.model_dir, n_best_to_keep=best_ckpts_topk)
                best_pairs = _list_best_ckpt_pairs(hps.model_dir)
                if best_pairs:
                    best_desc = ", ".join([f"step={step}, loss={loss:.6f}" for step, loss in best_pairs])
                    logger.info(f"Current best topk ({len(best_pairs)}/{best_ckpts_topk}): {best_desc}")
                else:
                    logger.info(f"Current best topk (0/{best_ckpts_topk}): <empty>")

            if global_step % checkpoint_save_interval == 0 and global_step != 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, discriminator, eval_loader, writer_eval):
    generator.eval()
    discriminator.eval()
    image_dict = {}
    audio_dict = {}
    eval_loss_sum = 0.0
    eval_batch_count = 0
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, lengths, uv, volume = items
            g = spk.cuda(0)
            spec, y = spec.cuda(0), y.cuda(0)
            c = c.cuda(0)
            f0 = f0.cuda(0)
            uv = uv.cuda(0)
            lengths = lengths.cuda(0)
            if volume is not None:
                volume = volume.cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = generator(c, f0, uv, spec,
                                                                                      g=g,
                                                                                      c_lengths=lengths,
                                                                                      spec_lengths=lengths,
                                                                                      vol=volume)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(y, y_hat)
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            use_auto_f0 = generator.module.use_automatic_f0_prediction if hasattr(generator, 'module') else generator.use_automatic_f0_prediction
            loss_lf0 = F.mse_loss(pred_lf0, lf0) if use_auto_f0 else loss_mel.new_tensor(0.0)
            eval_loss_g_total = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
            eval_loss_sum += float(eval_loss_g_total.item())
            eval_batch_count += 1

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })

    eval_loss_g_total_avg = eval_loss_sum / max(1, eval_batch_count)
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars={"loss/g/total": eval_loss_g_total_avg},
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()
    discriminator.train()
    return eval_loss_g_total_avg


if __name__ == "__main__":
    main()