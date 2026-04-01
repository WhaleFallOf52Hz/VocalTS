import argparse
import logging
import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

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

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from dataset_manifest import (
    collect_samples,
    ensure_parent,
    legacy_feature_path,
    new_feature_path,
)
import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def _target_path(sample, key):
    if sample.path_root is not None and sample.rel_path is not None:
        return new_feature_path(sample, key)
    return legacy_feature_path(sample.audio_path, key)


def process_one(sample, hmodel, device, diff=False, mel_extractor=None, f0_predictor=None):
    filename = sample.audio_path
    wav, sr = librosa.load(filename, sr=sampling_rate, mono=True)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = _target_path(sample, "units")
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        ensure_parent(soft_path)
        torch.save(c.cpu(), soft_path)

    f0_path = _target_path(sample, "f0")
    if not os.path.exists(f0_path):
        if f0_predictor is None:
            raise RuntimeError("f0_predictor is required when f0 file does not exist")
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        ensure_parent(f0_path)
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


    spec_path = _target_path(sample, "spec")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        #audio_norm = audio / hps.data.max_wav_value

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        ensure_parent(spec_path)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = _target_path(sample, "volume")
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            ensure_parent(volume_path)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = _target_path(sample, "mel")
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            ensure_parent(mel_path)
            np.save(mel_path, mel)
        aug_mel_path = _target_path(sample, "aug_mel")
        aug_vol_path = _target_path(sample, "aug_vol")
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            ensure_parent(aug_mel_path)
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            ensure_parent(aug_vol_path)
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


def process_batch(file_chunk, f0p, diff=False, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")

    f0_predictor = utils.get_f0_predictor(
        f0p,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        device=device,
        threshold=0.05,
    )
    logger.info(f"Loaded f0 predictor for rank {rank}")

    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    mel_extractor = None
    if diff:
        logger.info(f"Loading mel extractor for rank {rank} on {device}")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        logger.info(f"Loaded mel extractor for rank {rank}")
    for sample in tqdm(file_chunk, position = rank):
        process_one(sample, hmodel, device, diff, mel_extractor, f0_predictor)

def parallel_process(filenames, num_processes, f0p, diff, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Optional global split.json path"
    )
    parser.add_argument(
        "--split_train_key", type=str, default="train", help="split key for train set"
    )
    parser.add_argument(
        "--split_val_key", type=str, default="val", help="split key for validation set"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))

    if args.use_diff:
        print("use_diff")
        print("Mel extractor will be loaded per worker on its assigned device.")
    train_samples = collect_samples(
        source_dir=args.in_dir,
        split=args.split_train_key,
        extensions=["wav"],
        split_file=args.split_file,
    )
    val_samples = collect_samples(
        source_dir=args.in_dir,
        split=args.split_val_key,
        extensions=["wav"],
        split_file=args.split_file,
    )
    sample_map = {sample.audio_path: sample for sample in train_samples + val_samples}
    samples = list(sample_map.values())
    shuffle(samples)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(samples, num_processes, f0p, args.use_diff, device)
