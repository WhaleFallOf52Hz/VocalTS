#!/usr/bin/env python3
"""
Batch slice WAV/FLAC files into fixed-duration WAV chunks for DDSP-SVC training.
"""

import argparse
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


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
        required=True,
        help="Chunk duration in seconds, e.g. 2.0.",
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

    for wav_path in audio_files:
        try:
            info = sf.info(str(wav_path))
            slice_frames = int(round(args.slice_duration * info.samplerate))
            if slice_frames <= 0:
                raise ValueError("slice frame count must be positive")

            ranges = compute_slice_ranges(info.frames, slice_frames, args.discard_short)
            plans.append((wav_path, info, ranges))
            total_output_files += len(ranges)
            breakdown.append((wav_path, info.samplerate, info.frames, len(ranges)))
        except Exception as exc:
            failures.append((wav_path, str(exc)))

    print(f"Input directory: {input_dir}")
    print(f"Found audio files: {len(audio_files)}")
    print(f"Readable audio files: {len(plans)}")
    print(f"Unreadable audio files: {len(failures)}")
    print(f"Slice duration: {args.slice_duration:.3f} sec")
    print(f"Discard short tail: {args.discard_short}")
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

    for wav_path, info, ranges in tqdm(plans, desc="Slicing audio", unit="file"):
        if not ranges:
            continue

        audio, sample_rate = sf.read(str(wav_path), always_2d=False)
        for index, (start, end) in enumerate(ranges):
            chunk = audio[start:end]
            out_path = build_output_path(wav_path, input_dir, output_dir, index)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), chunk, sample_rate)
            written_files += 1

    print(f"\nDone. Wrote {written_files} wav files to: {output_dir}")


if __name__ == "__main__":
    main()
