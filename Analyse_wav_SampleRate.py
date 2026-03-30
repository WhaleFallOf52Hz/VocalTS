#!/usr/bin/env python3
"""
Batch inspect WAV sample rates in a folder and print summary statistics.
"""

import argparse
import collections
import wave
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect sample rates of WAV files in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing WAV files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file sample rate details.",
    )
    return parser.parse_args()


def collect_wav_files(input_dir: Path, recursive: bool):
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def read_sample_rate(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getframerate()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    wav_files = collect_wav_files(input_dir, args.recursive)
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")

    sample_rate_counter = collections.Counter()
    failures = []
    file_results = []

    for wav_path in wav_files:
        try:
            sample_rate = read_sample_rate(wav_path)
            sample_rate_counter[sample_rate] += 1
            file_results.append((wav_path, sample_rate))
        except Exception as exc:
            failures.append((wav_path, str(exc)))

    print(f"Input directory: {input_dir}")
    print(f"Scanned wav files: {len(wav_files)}")
    print(f"Successfully read: {sum(sample_rate_counter.values())}")
    print(f"Failed to read: {len(failures)}")

    if sample_rate_counter:
        print("\nSample rate distribution:")
        for sample_rate, count in sorted(sample_rate_counter.items()):
            print(f"  {sample_rate:>6} Hz: {count}")

        unique_rates = sorted(sample_rate_counter.keys())
        print("\nSummary:")
        print(f"  Unique sample rates: {len(unique_rates)}")
        print(f"  Min sample rate: {unique_rates[0]} Hz")
        print(f"  Max sample rate: {unique_rates[-1]} Hz")

    if args.show_files and file_results:
        print("\nPer-file sample rates:")
        for wav_path, sample_rate in file_results:
            print(f"  {wav_path}: {sample_rate} Hz")

    if failures:
        print("\nFailed files:")
        for wav_path, error in failures:
            print(f"  {wav_path}: {error}")


if __name__ == "__main__":
    main()
