import os
import random
import re
import json
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import concurrent.futures


def get_npy_shape(file_path):
    with open(file_path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape = np.lib.format.read_array_header_1_0(f)[0]
        elif version == (2, 0):
            shape = np.lib.format.read_array_header_2_0(f)[0]
        else:
            raise ValueError("Unsupported .npy file version")
    return shape
        
def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def normalize_rel_path(path):
    return path.replace('\\', '/').lstrip('./')


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


def load_split_file(path_root, train_key, val_key):
    split_path = os.path.join(path_root, 'split.json')
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f'split file not found: {split_path}')

    with open(split_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    train_entries = [str(v) for v in split_data.get(train_key, [])]
    val_entries = [str(v) for v in split_data.get(val_key, [])]
    return train_entries, val_entries


def resolve_split_entries(split_entries, available_files, path_root):
    available_files = [normalize_rel_path(v) for v in available_files]
    available_set = set(available_files)

    basename_map = {}
    for rel in available_files:
        basename = os.path.basename(rel)
        basename_map.setdefault(basename, []).append(rel)

    resolved = set()
    for entry in split_entries:
        entry_norm = normalize_rel_path(entry)
        if '/' in entry_norm:
            if entry_norm not in available_set:
                raise ValueError(f'split entry not found in audio folder ({path_root}): {entry_norm}')
            resolved.add(entry_norm)
            continue

        candidates = basename_map.get(entry_norm, [])
        if len(candidates) == 1:
            resolved.add(candidates[0])
        elif len(candidates) == 0:
            raise ValueError(f'split entry not found in audio folder ({path_root}): {entry_norm}')
        else:
            raise ValueError(
                f'ambiguous split entry in {path_root}: {entry_norm}, matched {candidates}. '
                'Use relative path instead of basename.'
            )

    return sorted(resolved)


def collect_split_samples(data_root, extensions, train_key='train', val_key='val'):
    path_roots = discover_data_roots(data_root)
    if len(path_roots) == 0:
        raise FileNotFoundError(f'no valid data roots with audio folder found under: {data_root}')

    sample_dict = {'train': [], 'val': []}
    for path_root in path_roots:
        audio_root = os.path.join(path_root, 'audio')
        available_files = traverse_dir(
            audio_root,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        if len(available_files) == 0:
            print(f'Warning: no audio files in {audio_root}, skipped.')
            continue

        train_entries, val_entries = load_split_file(path_root, train_key, val_key)
        train_files = resolve_split_entries(train_entries, available_files, path_root)
        val_files = resolve_split_entries(val_entries, available_files, path_root)

        overlap = set(train_files) & set(val_files)
        if len(overlap) > 0:
            raise ValueError(f'split overlap in {path_root}: {sorted(overlap)[:10]}')

        for name_ext in train_files:
            sample_dict['train'].append((path_root, name_ext))
        for name_ext in val_files:
            sample_dict['val'].append((path_root, name_ext))

    if len(sample_dict['train']) == 0:
        raise ValueError('no train samples found from split.json files')
    if len(sample_dict['val']) == 0:
        raise ValueError('no val samples found from split.json files')

    return sample_dict


def get_data_loaders(args, whole_audio=False):
    train_key = args.data.split_train_key if args.data.split_train_key is not None else 'train'
    val_key = args.data.split_val_key if args.data.split_val_key is not None else 'val'
    samples = collect_split_samples(
        args.data.root_path,
        extensions=args.data.extensions,
        train_key=train_key,
        val_key=val_key
    )

    data_train = AudioDataset(
        samples['train'],
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    data_valid = AudioDataset(
        samples['val'],
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        samples,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.samples = list(samples)
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}

        self.pitch_aug_dicts = {}
        unique_roots = sorted({path_root for path_root, _ in self.samples})
        for path_root in unique_roots:
            path_pitch = os.path.join(path_root, 'pitch_aug_dict.npy')
            if os.path.isfile(path_pitch):
                self.pitch_aug_dicts[path_root] = np.load(path_pitch, allow_pickle=True).item()
            else:
                self.pitch_aug_dicts[path_root] = {}

        if load_all_data:
            print('Load all the data from split samples')
        else:
            print('Load the f0, volume data from split samples')
        
        def _load_single_file(sample_idx, sample):
            path_root, name_ext = sample
            name = os.path.splitext(name_ext)[0]

            path_f0 = os.path.join(path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0_len = len(f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

            path_volume = os.path.join(path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume_len = len(volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

            path_augvol = os.path.join(path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol_len = len(aug_vol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)

            if n_spk is not None and n_spk > 1:
                dirname_split = os.path.basename(path_root)
                if not str(dirname_split).isdigit():
                    dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(' [x] Muiti-speaker training error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            path_mel = os.path.join(path_root, 'mel', name_ext) + '.npy'
            path_augmel = os.path.join(path_root, 'aug_mel', name_ext) + '.npy'
            path_units = os.path.join(path_root, 'units', name_ext) + '.npy'

            mel_len = get_npy_shape(path_mel)[0]
            aug_mel_len = get_npy_shape(path_augmel)[0]
            units_len = get_npy_shape(path_units)[0]
            frame_len = min(mel_len, aug_mel_len, units_len, f0_len, volume_len, aug_vol_len)

            if load_all_data:
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)

                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)

                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)

                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()

                data_dict = {
                        'frame_len': frame_len,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id,
                        'path_root': path_root,
                        'name_ext': name_ext
                        }
            else:
                data_dict = {
                        'frame_len': frame_len,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id,
                        'path_root': path_root,
                        'name_ext': name_ext
                        }
            return sample_idx, data_dict

        max_workers = max(min(32, os.cpu_count()), 4)
        print(f'Using {max_workers} workers for parallel data loading')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(_load_single_file, sample_idx, sample): sample
                for sample_idx, sample in enumerate(self.samples)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(self.samples), desc='Loading data'):
                try:
                    sample_idx, data_dict = future.result()
                    self.data_buffer[sample_idx] = data_dict
                except Exception as e:
                    print(f'Error loading {future_to_name[future]}: {e}')
                    raise

    def __getitem__(self, file_idx):
        data_buffer = self.data_buffer[file_idx]
        # check duration. if too short, then skip
        if data_buffer['frame_len'] < self.crop_len:
            return self.__getitem__( (file_idx + 1) % len(self.samples))
            
        # get item
        return self.get_data(data_buffer)

    def get_data(self, data_buffer):
        path_root = data_buffer['path_root']
        name_ext = data_buffer['name_ext']
        name = os.path.splitext(name_ext)[0]
        display_name = normalize_rel_path(os.path.join(os.path.basename(path_root), name))
        start_frame = 0 if self.whole_audio else random.randint(0, data_buffer['frame_len'] - self.crop_len)
        units_frame_len = data_buffer['frame_len'] if self.whole_audio else self.crop_len
        aug_flag = random.choice([True, False]) and self.use_aug

        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel, mmap_mode='r')
            mel = mel[start_frame : start_frame + units_frame_len].copy()
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(path_root, 'units', name_ext) + '.npy'
            units = np.load(units, mmap_mode='r')
            units = units[start_frame : start_frame + units_frame_len].copy()
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dicts.get(path_root, {}).get(name_ext, 0.0)
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()

        path_audio = os.path.join(path_root, 'audio', name_ext)
        
        return dict(
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            aug_shift=aug_shift,
            name=display_name,
            name_ext=name_ext,
            path_audio=path_audio
        )

    def __len__(self):
        return len(self.samples)
