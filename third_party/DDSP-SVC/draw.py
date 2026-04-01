import argparse
import json
import os
import random
import shutil

import numpy as np
import soundfile as sf
import tqdm


WAV_MIN_LENGTH = 2
SAMPLE_MIN = 2
SAMPLE_MAX = 10


def parse_args(args=None, namespace=None):
    root_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="split",
        choices=["split", "move"],
        help="split: generate split.json, move: legacy move files from train to val"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=root_dir + "/data",
        help="data root for split mode"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="split.json",
        help="split filename under each speaker/data root"
    )
    parser.add_argument(
        "--train-key",
        type=str,
        default="train",
        help="train key in split json"
    )
    parser.add_argument(
        "--val-key",
        type=str,
        default="val",
        help="val key in split json"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed"
    )

    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default=root_dir + "/data/train/audio",
        help="directory where contains train dataset (move mode)"
    )
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=root_dir + "/data/val/audio",
        help="directory where contains validate dataset (move mode)"
    )
    parser.add_argument(
        "-r",
        "--sample_rate",
        type=float,
        default=1,
        help="The percentage of files to be extracted"
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        required=False,
        nargs="*",
        default=["wav", "flac"],
        help="list of using file extensions, e.g.) -f wav flac ..."
    )
    return parser.parse_args(args=args, namespace=namespace)


def check_duration(wav_file):
    f = sf.SoundFile(wav_file)
    frames = len(f)
    rate = f.samplerate
    duration = frames / float(rate)
    f.close()
    return duration > WAV_MIN_LENGTH


def discover_data_roots(data_root):
    roots = []
    root_audio = os.path.join(data_root, 'audio')
    if os.path.isdir(root_audio):
        roots.append(data_root)

    if not os.path.isdir(data_root):
        raise NotADirectoryError(f'data root does not exist: {data_root}')

    for item in sorted(os.listdir(data_root)):
        candidate = os.path.join(data_root, item)
        if not os.path.isdir(candidate):
            continue
        if os.path.isdir(os.path.join(candidate, 'audio')):
            roots.append(candidate)

    unique_roots = []
    seen = set()
    for root in roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)
    return unique_roots


def list_audio_files(audio_root, extensions):
    files = []
    for root, _, filenames in os.walk(audio_root):
        for filename in filenames:
            if any(filename.endswith(f".{ext}") for ext in extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, audio_root).replace('\\', '/')
                files.append(rel_path)
    files.sort()
    return files


def resolve_unique_basenames(rel_paths, path_root):
    basename_map = {}
    for rel_path in rel_paths:
        basename = os.path.basename(rel_path)
        basename_map.setdefault(basename, []).append(rel_path)

    duplicated = {k: v for k, v in basename_map.items() if len(v) > 1}
    if len(duplicated) > 0:
        first_key = sorted(duplicated.keys())[0]
        raise ValueError(
            f'Found duplicated basename in {path_root}: {first_key} -> {duplicated[first_key]}. '
            'split.json uses basename in this mode, please make filenames unique per speaker root.'
        )

    return basename_map


def generate_split_for_root(path_root, ratio, extensions, split_file, train_key, val_key):
    audio_root = os.path.join(path_root, 'audio')
    rel_paths = list_audio_files(audio_root, extensions)
    if len(rel_paths) == 0:
        print(f"Warning: no audio files found in {audio_root}, skipped.")
        return

    basename_map = resolve_unique_basenames(rel_paths, path_root)

    valid_basenames = []
    for basename, rel_path_list in basename_map.items():
        rel_path = rel_path_list[0]
        src_file = os.path.join(audio_root, rel_path)
        if not check_duration(src_file):
            print(f"Skipped {src_file} because its duration is less than {WAV_MIN_LENGTH} seconds.")
            continue
        valid_basenames.append(basename)

    valid_basenames.sort()
    if len(valid_basenames) == 0:
        print(f"Warning: no valid audio files in {audio_root}, skipped.")
        return

    num_files = int(len(valid_basenames) * ratio)
    num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))
    num_files = min(num_files, len(valid_basenames))

    random.shuffle(valid_basenames)
    val_list = sorted(valid_basenames[:num_files])
    val_set = set(val_list)
    train_list = sorted([f for f in valid_basenames if f not in val_set])

    split_path = os.path.join(path_root, split_file)
    split_data = {
        train_key: train_list,
        val_key: val_list
    }
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)
    print(f"Saved split file: {split_path} | train={len(train_list)} val={len(val_list)}")


def split_data_move_mode(src_dir, dst_dir, ratio, extensions):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    subdirs, files, subfiles = [], [], []
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isfile(subitem_path) and any([subitem.endswith(f".{ext}") for ext in extensions]):
                    subfiles.append(subitem)
        elif os.path.isfile(item_path) and any([item.endswith(f".{ext}") for ext in extensions]):
            files.append(item)

    if len(files) == 0 and len(subfiles) == 0:
        print(f"Error: No wav files found in {src_dir}")
        return

    num_files = int(len(files) * ratio)
    num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))
    num_files = min(num_files, len(files))

    np.random.shuffle(files)
    selected_files = files[:num_files]

    pbar = tqdm.tqdm(total=num_files)
    for file in selected_files:
        src_file = os.path.join(src_dir, file)
        if not check_duration(src_file):
            print(f"Skipped {src_file} because its duration is less than {WAV_MIN_LENGTH} seconds.")
            continue
        dst_file = os.path.join(dst_dir, file)
        shutil.move(src_file, dst_file)
        pbar.update(1)
    pbar.close()

    for subdir in subdirs:
        src_subdir = os.path.join(src_dir, subdir)
        dst_subdir = os.path.join(dst_dir, subdir)
        split_data_move_mode(src_subdir, dst_subdir, ratio, extensions)


def run_split_mode(cmd):
    random.seed(cmd.seed)
    np.random.seed(cmd.seed)

    ratio = cmd.sample_rate / 100
    path_roots = discover_data_roots(cmd.data_root)
    if len(path_roots) == 0:
        raise FileNotFoundError(f'no valid data roots with audio folder found under: {cmd.data_root}')

    for path_root in path_roots:
        generate_split_for_root(
            path_root,
            ratio=ratio,
            extensions=cmd.extensions,
            split_file=cmd.split_file,
            train_key=cmd.train_key,
            val_key=cmd.val_key,
        )


def run_move_mode(cmd):
    ratio = cmd.sample_rate / 100
    split_data_move_mode(cmd.train, cmd.val, ratio, cmd.extensions)


def main(cmd):
    if cmd.mode == 'split':
        run_split_mode(cmd)
    else:
        run_move_mode(cmd)


if __name__ == "__main__":
    cmd = parse_args()
    main(cmd)
