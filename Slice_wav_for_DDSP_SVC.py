#!/usr/bin/env python3
"""
Batch slice WAV/FLAC files into fixed-duration WAV chunks for DDSP-SVC training.

By default, the script first removes long silent gaps with the bundled
audio-slicer and then packs the remaining voiced audio into ~10 second chunks.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SLICER_DIR = SCRIPT_DIR / "third_party" / "audio-slicer"
if SLICER_DIR.is_dir() and str(SLICER_DIR) not in sys.path:
    sys.path.insert(0, str(SLICER_DIR))

try:
    from slicer2 import Slicer
except Exception:
    Slicer = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice WAV/FLAC files in a folder into fixed-duration WAV chunks for DDSP-SVC."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing source wav/flac files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save sliced wav files.",
    )
    parser.add_argument(
        "--slice-duration",
        type=float,
        default=10.0,
        help="Chunk duration in seconds, default: 10.0.",
    )
    parser.add_argument(
        "--discard-short",
        action="store_true",
        help="Discard the final chunk if it is shorter than --slice-duration.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview how many output files will be produced without writing files.",
    )
    parser.add_argument(
        "--show-breakdown",
        action="store_true",
        help="Print per-file slice counts in the summary.",
    )
    parser.add_argument(
        "--no-silence-aware",
        dest="silence_aware",
        action="store_false",
        help="Disable silence-aware packing and use fixed-duration slicing only.",
    )
    parser.set_defaults(silence_aware=True)
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-40.0,
        help="RMS threshold in dB for silence detection. Default: -40.0.",
    )
    parser.add_argument(
        "--silence-min-length-ms",
        type=int,
        default=5000,
        help="Minimum voiced length before a silence break can be considered. Default: 5000.",
    )
    parser.add_argument(
        "--silence-min-interval-ms",
        type=int,
        default=300,
        help="Minimum silent interval that can trigger a cut. Default: 300.",
    )
    parser.add_argument(
        "--silence-hop-ms",
        type=int,
        default=10,
        help="RMS frame hop size in milliseconds. Default: 10.",
    )
    parser.add_argument(
        "--silence-max-kept-ms",
        type=int,
        default=500,
        help="Maximum silence kept around a slice boundary. Default: 500.",
    )
    return parser.parse_args()


def collect_audio_files(input_dir: Path, recursive: bool):
    patterns = ("**/*.wav", "**/*.flac") if recursive else ("*.wav", "*.flac")
    files = []
    for pattern in patterns:
        files.extend(path for path in input_dir.glob(pattern) if path.is_file())
    return sorted(files)


def compute_slice_ranges(total_frames: int, slice_frames: int, discard_short: bool):
    ranges = []
    start = 0
    while start < total_frames:
        end = min(start + slice_frames, total_frames)
        if end - start < slice_frames and discard_short:
            break
        ranges.append((start, end))
        start += slice_frames
    return ranges


def load_audio(path: Path):
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def build_silence_slicer(sample_rate: int, args):
    if Slicer is None:
        raise RuntimeError(
            "Silence-aware mode requires third_party/audio-slicer/slicer2.py, "
            "but it could not be imported."
        )
    return Slicer(
        sr=sample_rate,
        threshold=args.silence_threshold_db,
        min_length=args.silence_min_length_ms,
        min_interval=args.silence_min_interval_ms,
        hop_size=args.silence_hop_ms,
        max_sil_kept=args.silence_max_kept_ms,
    )


def silence_pack_audio(audio, sample_rate: int, args):
    slicer = build_silence_slicer(sample_rate, args)
    chunks = slicer.slice(audio)
    voiced = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks if len(chunk) > 0]
    if not voiced:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(voiced, axis=0)


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path, index: int):
    relative_path = input_path.relative_to(input_dir)
    relative_parent = relative_path.parent
    stem = input_path.stem
    output_name = f"{stem}_{index:04d}.wav"
    return output_dir / relative_parent / output_name


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.slice_duration <= 0:
        raise ValueError("--slice-duration must be greater than 0.")

    audio_files = collect_audio_files(input_dir, args.recursive)
    if not audio_files:
        raise FileNotFoundError(f"No .wav or .flac files found in: {input_dir}")

    total_output_files = 0
    failures = []
    breakdown = []
    plans = []

    for wav_path in tqdm(audio_files):
        try:
            audio, sample_rate = load_audio(wav_path)
            if args.silence_aware:
                audio = silence_pack_audio(audio, sample_rate, args)
            if audio.size == 0:
                raise ValueError("No voiced audio left after silence-aware slicing")

            slice_frames = int(round(args.slice_duration * sample_rate))
            if slice_frames <= 0:
                raise ValueError("slice frame count must be positive")

            ranges = compute_slice_ranges(len(audio), slice_frames, args.discard_short)
            plans.append((wav_path, sample_rate, audio, ranges))
            total_output_files += len(ranges)
            breakdown.append((wav_path, sample_rate, len(audio), len(ranges)))
        except Exception as exc:
            failures.append((wav_path, str(exc)))

    print(f"Input directory: {input_dir}")
    print(f"Found audio files: {len(audio_files)}")
    print(f"Readable audio files: {len(plans)}")
    print(f"Unreadable audio files: {len(failures)}")
    print(f"Slice duration: {args.slice_duration:.3f} sec")
    print(f"Discard short tail: {args.discard_short}")
    print(f"Silence aware: {args.silence_aware}")
    print(f"Projected output files: {total_output_files}")

    if args.show_breakdown:
        print("\nPer-file slice counts:")
        for wav_path, sample_rate, frames, count in breakdown:
            duration = frames / float(sample_rate)
            print(f"  {wav_path}: {count} slices, {duration:.3f} sec, {sample_rate} Hz")

    if failures:
        print("\nUnreadable files:")
        for wav_path, error in failures:
            print(f"  {wav_path}: {error}")

    if args.dry_run:
        print("\nDry run only. No files were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files = 0

    for wav_path, sample_rate, audio, ranges in tqdm(plans, desc="Slicing audio", unit="file"):
        if not ranges:
            continue

        for index, (start, end) in enumerate(ranges):
            chunk = audio[start:end]
            out_path = build_output_path(wav_path, input_dir, output_dir, index)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), chunk, sample_rate)
            written_files += 1

    print(f"\nDone. Wrote {written_files} wav files to: {output_dir}")


if __name__ == "__main__":
    main()
