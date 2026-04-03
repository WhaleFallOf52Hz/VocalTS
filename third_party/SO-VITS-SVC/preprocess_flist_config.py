import argparse
import json
import os
import re
import wave
from random import shuffle

from loguru import logger
from tqdm import tqdm

from dataset_manifest import discover_data_roots, normalize_rel_path
import diffusion.logger.utils as du

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="legacy path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="legacy path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    parser.add_argument("--split_train_key", type=str, default="train", help="split key for train set")
    parser.add_argument("--split_val_key", type=str, default="val", help="split key for validation set")
    parser.add_argument("--val_count", type=int, default=2, help="default validation clips per speaker/data root")
    parser.add_argument("--write_filelists", action="store_true", help="also export legacy filelists/train.txt and filelists/val.txt")
    args = parser.parse_args()
    
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    spk_dict = {}
    spk_id = 0

    roots = discover_data_roots(args.source_dir, ["wav"])
    if not roots:
        raise FileNotFoundError(f"No valid audio data roots found under: {args.source_dir}")

    for path_root, audio_root, speaker in tqdm(roots):
        if speaker not in spk_dict:
            spk_dict[speaker] = spk_id
            spk_id += 1

        wavs = []
        for root, _, files in os.walk(audio_root):
            for file_name in files:
                if not file_name.endswith("wav"):
                    continue
                if file_name.startswith("."):
                    continue

                file_path = os.path.join(root, file_name)
                rel_path = normalize_rel_path(os.path.relpath(file_path, audio_root))

                if not pattern.match(rel_path):
                    logger.warning("Detected non-ASCII file name: " + file_path)

                if get_wav_duration(file_path) < 0.3:
                    logger.info("Skip too short audio: " + file_path)
                    continue

                wavs.append((file_path.replace("\\", "/"), rel_path))

        shuffle(wavs)
        val_count = min(max(args.val_count, 0), len(wavs))
        val_local = wavs[:val_count]
        train_local = wavs[val_count:]
        if len(train_local) == 0 and len(val_local) > 0:
            train_local = val_local[-1:]
            val_local = val_local[:-1]

        split_json_path = os.path.join(path_root, "split.json")
        split_content = {
            args.split_train_key: [item[1] for item in train_local],
            args.split_val_key: [item[1] for item in val_local],
        }
        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump(split_content, f, indent=2, ensure_ascii=False)

        train += [item[0] for item in train_local]
        val += [item[0] for item in val_local]

    shuffle(train)
    shuffle(val)

    if args.write_filelists:
        logger.info("Writing " + args.train_list)
        os.makedirs(os.path.dirname(args.train_list), exist_ok=True)
        with open(args.train_list, "w") as f:
            for fname in tqdm(train):
                f.write(fname + "\n")

        logger.info("Writing " + args.val_list)
        os.makedirs(os.path.dirname(args.val_list), exist_ok=True)
        with open(args.val_list, "w") as f:
            for fname in tqdm(val):
                f.write(fname + "\n")


    d_config_template = du.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id
    d_config_template["data"]["encoder"] = args.speech_encoder
    d_config_template["spk"] = spk_dict
    d_config_template["data"]["source_dir"] = args.source_dir
    d_config_template["data"]["split_file"] = ""
    d_config_template["data"]["split_train_key"] = args.split_train_key
    d_config_template["data"]["split_val_key"] = args.split_val_key
    
    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
    config_template["model"]["speech_encoder"] = args.speech_encoder
    config_template["train"]["save_interval"] = 10000
    config_template["train"]["keep_ckpts"] = 1
    config_template["data"]["source_dir"] = args.source_dir
    config_template["data"]["split_file"] = ""
    config_template["data"]["split_train_key"] = args.split_train_key
    config_template["data"]["split_val_key"] = args.split_val_key
    
    if args.speech_encoder == "vec768l12" or args.speech_encoder == "dphubert" or args.speech_encoder == "wavlmbase+":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 768
        d_config_template["data"]["encoder_out_channels"] = 768
    elif args.speech_encoder == "vec256l9" or args.speech_encoder == 'hubertsoft':
        config_template["model"]["ssl_dim"] = config_template["model"]["gin_channels"] = 256
        d_config_template["data"]["encoder_out_channels"] = 256
    elif args.speech_encoder == "whisper-ppg" or args.speech_encoder == 'cnhubertlarge':
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1024
        d_config_template["data"]["encoder_out_channels"] = 1024
    elif args.speech_encoder == "whisper-ppg-large":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1280
        d_config_template["data"]["encoder_out_channels"] = 1280
        
    if args.vol_aug:
        config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True

    if args.tiny:
        config_template["model"]["filter_channels"] = 512

    if args.write_filelists:
        config_template["data"]["training_files"] = args.train_list
        config_template["data"]["validation_files"] = args.val_list
        d_config_template["data"]["training_files"] = args.train_list
        d_config_template["data"]["validation_files"] = args.val_list

    logger.info("Writing to configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    logger.info("Writing to configs/diffusion.yaml")
    du.save_config("configs/diffusion.yaml",d_config_template)
