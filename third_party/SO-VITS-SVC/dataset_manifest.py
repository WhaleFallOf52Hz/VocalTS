import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SampleItem:
    audio_path: str
    speaker: str
    path_root: Optional[str] = None
    rel_path: Optional[str] = None


def normalize_rel_path(path: str) -> str:
    return path.replace('\\', '/').lstrip('./')


def _list_audio_files(audio_dir: str, extensions: Sequence[str]) -> List[str]:
    exts = tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)
    files: List[str] = []
    for root, _, names in os.walk(audio_dir):
        for name in names:
            if name.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(root, name), audio_dir)
                files.append(normalize_rel_path(rel))
    files.sort()
    return files


def discover_data_roots(base_dir: str, extensions: Sequence[str]) -> List[Tuple[str, str, str]]:
    base_dir = os.path.abspath(base_dir)
    roots: List[Tuple[str, str, str]] = []

    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"Data directory does not exist: {base_dir}")

    single_audio = os.path.join(base_dir, 'audio')
    if os.path.isdir(single_audio):
        roots.append((base_dir, single_audio, Path(base_dir).name or 'default'))

    for name in sorted(os.listdir(base_dir)):
        child = os.path.join(base_dir, name)
        if not os.path.isdir(child):
            continue
        child_audio = os.path.join(child, 'audio')
        if os.path.isdir(child_audio):
            roots.append((child, child_audio, name))

    if roots:
        return roots

    direct_audio = _list_audio_files(base_dir, extensions)
    if direct_audio:
        return [(base_dir, base_dir, Path(base_dir).name or 'default')]

    for name in sorted(os.listdir(base_dir)):
        child = os.path.join(base_dir, name)
        if not os.path.isdir(child):
            continue
        child_files = _list_audio_files(child, extensions)
        if child_files:
            roots.append((child, child, name))

    return roots


def _resolve_split_entries(entries: Sequence[str], available_files: Sequence[str], split_path: str) -> List[str]:
    available_files = [normalize_rel_path(v) for v in available_files]
    available_set = set(available_files)

    basename_map: Dict[str, List[str]] = {}
    for rel in available_files:
        basename = os.path.basename(rel)
        basename_map.setdefault(basename, []).append(rel)

    resolved: List[str] = []
    for entry in entries:
        entry_norm = normalize_rel_path(str(entry))
        if '/' in entry_norm:
            if entry_norm not in available_set:
                raise ValueError(f"split entry not found ({split_path}): {entry_norm}")
            resolved.append(entry_norm)
            continue

        candidates = basename_map.get(entry_norm, [])
        if len(candidates) == 1:
            resolved.append(candidates[0])
        elif len(candidates) == 0:
            raise ValueError(f"split entry not found ({split_path}): {entry_norm}")
        else:
            raise ValueError(
                f"ambiguous split entry ({split_path}): {entry_norm}, matched {candidates}. "
                "Use relative path from audio root instead of basename."
            )
    return resolved


def _load_split_json(split_json_path: str, split_key: str) -> List[str]:
    with open(split_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if split_key not in data:
        raise KeyError(f"split key '{split_key}' not found in {split_json_path}")
    return [str(v) for v in data.get(split_key, [])]


def collect_samples(
    source_dir: str,
    split: str,
    extensions: Sequence[str],
    split_file: Optional[str] = None,
) -> List[SampleItem]:
    source_dir = os.path.abspath(source_dir)

    if split_file:
        split_file = os.path.abspath(split_file)
        split_entries = _load_split_json(split_file, split)
        samples: List[SampleItem] = []
        for entry in split_entries:
            entry_path = entry if os.path.isabs(entry) else os.path.join(source_dir, entry)
            entry_path = os.path.abspath(entry_path)
            speaker = Path(entry_path).parent.name
            samples.append(SampleItem(audio_path=entry_path, speaker=speaker))
        return samples

    roots = discover_data_roots(source_dir, extensions)
    if not roots:
        raise FileNotFoundError(f"No audio roots discovered under: {source_dir}")

    samples: List[SampleItem] = []
    for path_root, audio_root, speaker in roots:
        available = _list_audio_files(audio_root, extensions)
        if not available:
            continue
        split_json = os.path.join(path_root, 'split.json')
        if not os.path.isfile(split_json):
            raise FileNotFoundError(f"split.json not found: {split_json}")

        entries = _load_split_json(split_json, split)
        rel_paths = _resolve_split_entries(entries, available, split_json)
        for rel in rel_paths:
            audio_path = os.path.join(audio_root, rel)
            samples.append(
                SampleItem(
                    audio_path=os.path.abspath(audio_path),
                    speaker=speaker,
                    path_root=os.path.abspath(path_root),
                    rel_path=normalize_rel_path(rel),
                )
            )

    if not samples:
        raise ValueError(f"No '{split}' samples found from split.json under: {source_dir}")
    return samples


def load_legacy_filelist(filelist_path: str) -> List[SampleItem]:
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    samples: List[SampleItem] = []
    for path in lines:
        speaker = Path(path).parent.name
        samples.append(SampleItem(audio_path=path, speaker=speaker))
    return samples


def new_feature_path(sample: SampleItem, key: str) -> str:
    if sample.path_root is None or sample.rel_path is None:
        raise ValueError('new_feature_path requires split-based sample with path_root and rel_path')

    rel = sample.rel_path
    rel_no_ext = os.path.splitext(rel)[0]
    root = sample.path_root

    mapping = {
        'units': os.path.join(root, 'units', rel + '.pt'),
        'f0': os.path.join(root, 'f0', rel + '.npy'),
        'spec': os.path.join(root, 'spec', rel_no_ext + '.pt'),
        'volume': os.path.join(root, 'volume', rel + '.npy'),
        'mel': os.path.join(root, 'mel', rel + '.npy'),
        'aug_mel': os.path.join(root, 'aug_mel', rel + '.npy'),
        'aug_vol': os.path.join(root, 'aug_vol', rel + '.npy'),
    }
    if key not in mapping:
        raise KeyError(f'Unknown feature key: {key}')
    return mapping[key]


def legacy_feature_path(audio_path: str, key: str) -> str:
    mapping = {
        'units': audio_path + '.soft.pt',
        'f0': audio_path + '.f0.npy',
        'spec': os.path.splitext(audio_path)[0] + '.spec.pt',
        'volume': audio_path + '.vol.npy',
        'mel': audio_path + '.mel.npy',
        'aug_mel': audio_path + '.aug_mel.npy',
        'aug_vol': audio_path + '.aug_vol.npy',
    }
    if key not in mapping:
        raise KeyError(f'Unknown feature key: {key}')
    return mapping[key]


def resolve_feature_read_path(sample: SampleItem, key: str) -> str:
    if sample.path_root is not None and sample.rel_path is not None:
        new_path = new_feature_path(sample, key)
        if os.path.exists(new_path):
            return new_path
    return legacy_feature_path(sample.audio_path, key)


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
