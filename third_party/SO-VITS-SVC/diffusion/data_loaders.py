import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_manifest import collect_samples, load_legacy_filelist, resolve_feature_read_path
from utils import repeat_expand_2d


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


def get_data_loaders(args, whole_audio=False):
    source_dir = getattr(args.data, 'source_dir', None)
    split_file = getattr(args.data, 'split_file', None)
    train_source = source_dir if source_dir else args.data.training_files
    val_source = source_dir if source_dir else args.data.validation_files

    data_train = AudioDataset(
        data_source=train_source,
        split='train',
        split_file=split_file if split_file else None,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        spk=args.spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
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
        data_source=val_source,
        split='val',
        split_file=split_file if split_file else None,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        spk=args.spk,
        extensions=args.data.extensions,
        unit_interpolate_mode = args.data.unit_interpolate_mode,
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
        data_source,
        split,
        waveform_sec,
        hop_size,
        sample_rate,
        spk,
        split_file=None,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        unit_interpolate_mode = 'left'
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.data_source = data_source
        self.split = split
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = {}
        self.unit_interpolate_mode = unit_interpolate_mode
        self.min_duration = self.waveform_sec
        self.valid_sample_indices = []
        # np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if os.path.isdir(data_source):
            self.samples = collect_samples(
                source_dir=data_source,
                split=split,
                extensions=extensions,
                split_file=split_file if split_file else None,
            )
        else:
            self.samples = load_legacy_filelist(data_source)

        if load_all_data:
            print('Load all the data split:', split)
        else:
            print('Load the f0, volume data split:', split)

        for sample_idx, sample in tqdm(enumerate(self.samples), total=len(self.samples)):
            path_audio = sample.audio_path
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            
            path_f0 = resolve_feature_read_path(sample, 'f0')
            f0,_ = np.load(path_f0,allow_pickle=True)
            f0 = torch.from_numpy(np.array(f0,dtype=float)).float().unsqueeze(-1).to(device)
                
            path_volume = resolve_feature_read_path(sample, 'volume')
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = resolve_feature_read_path(sample, 'aug_vol')
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                        
            if n_spk is not None and n_spk > 1:
                spk_name = sample.speaker
                spk_id = spk[spk_name] if spk_name in spk else 0
                if spk_id < 0 or spk_id >= n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be a positive integer from 0 to n_spk-1 ')
            else:
                spk_id = 0
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = resolve_feature_read_path(sample, 'mel')
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)
                
                path_augmel = resolve_feature_read_path(sample, 'aug_mel')
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                aug_mel = np.array(aug_mel,dtype=float)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                self.pitch_aug_dict[sample_idx] = keyshift

                path_units = resolve_feature_read_path(sample, 'units')
                units = torch.load(path_units).to(device)
                units = units[0]  
                units = repeat_expand_2d(units,f0.size(0),unit_interpolate_mode).transpose(0,1)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    
                self.data_buffer[sample_idx] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                    'spk_id': spk_id,
                    'sample': sample,
                        }
            else:
                path_augmel = resolve_feature_read_path(sample, 'aug_mel')
                aug_mel,keyshift = np.load(path_augmel, allow_pickle=True)
                self.pitch_aug_dict[sample_idx] = keyshift
                self.data_buffer[sample_idx] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                    'spk_id': spk_id,
                    'sample': sample,
                        }

        if self.whole_audio:
            self.valid_sample_indices = list(range(len(self.samples)))
        else:
            self.valid_sample_indices = [
                sample_idx
                for sample_idx, data_buffer in self.data_buffer.items()
                if data_buffer['duration'] >= self.min_duration
            ]

        if len(self.valid_sample_indices) == 0:
            raise ValueError(
                f"No valid samples found for split={self.split}. "
                f"Required duration >= {self.min_duration:.2f}s, "
                f"but all {len(self.samples)} samples are shorter. "
                "Please reduce data.duration in config or use longer audio clips."
            )
           

    def __getitem__(self, file_idx):
        sample_idx = self.valid_sample_indices[file_idx]
        data_buffer = self.data_buffer[sample_idx]
            
        # get item
        return self.get_data(sample_idx, data_buffer)

    def get_data(self, file_idx, data_buffer):
        sample = data_buffer['sample']
        audio_path = sample.audio_path
        name = os.path.splitext(audio_path)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        max_offset = max(duration - waveform_sec, 0.0)
        idx_from = 0 if self.whole_audio or max_offset <= 0 else random.uniform(0, max_offset)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        '''
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        '''
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = resolve_feature_read_path(sample, 'mel')
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[file_idx]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load units
        units = data_buffer.get('units')
        if units is None:
            path_units = resolve_feature_read_path(sample, 'units')
            units = torch.load(path_units)
            units = units[0]  
            units = repeat_expand_2d(units,f0.size(0),self.unit_interpolate_mode).transpose(0,1)
            
        units = units[start_frame : start_frame + units_frame_len]

        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=audio_path)

    def __len__(self):
        return len(self.valid_sample_indices)