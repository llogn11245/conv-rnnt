import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
import librosa
from glob import glob
import os

# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def __len__(self):
        return len(self.vocab)

def compute_gmvn(voice_path, sample_rate=16000):
    wav_files = glob(os.path.join(voice_path, "**", "*.wav"), recursive=True)

    win_length = int(0.025 * sample_rate)   # 25ms = 400 samples
    hop_length = int(0.010 * sample_rate)   # 10ms = 160 samples
    
    sum_feats = torch.zeros(192)
    sum_squares = torch.zeros(192)
    total_frames = 0

    for file in tqdm(wav_files):  # dataset là list/tập của waveform tensors
        with torch.no_grad():
            waveform, _ = librosa.load(file, sr=sample_rate)
            stft = librosa.stft(waveform, n_fft=512, win_length=win_length, hop_length=hop_length)
            mag = np.abs(stft[:64, :])  # Lấy 64 bins đầu tiên (low frequencies)
            log_mag = np.log1p(mag)  # log(1 + x)
            log_mag = log_mag.T  # [T, 64]

            stacked_feats = []
            for i in range(len(log_mag) - 6):  # skip rate = 3
                if i % 3 == 0:
                    stacked = np.concatenate([log_mag[i], log_mag[i+3], log_mag[i+6]])  # [192]
                    stacked_feats.append(stacked)

            stacked_feats = torch.tensor(np.array(stacked_feats), dtype=torch.float32)  # [T', 192]

            total_frames += stacked_feats.shape[0]
            sum_feats += stacked_feats.sum(dim=0)
            sum_squares += (stacked_feats ** 2).sum(dim=0)

    mean = sum_feats / total_frames
    std = (sum_squares / total_frames - mean**2).sqrt()
    return mean, std

class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, gmvn_mean = None, gmvn_std = None):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        
        self.gmvn_mean = gmvn_mean
        self.gmvn_std = gmvn_std
        # stats = torch.load(cmvn_stats) 
        # self.cmvn_mean = stats['mean']
        # self.cmvn_std = stats['std']
            
    def __len__(self):
        return len(self.data)

    def extract_features(self, wav_file, sr=16000):
        # Load waveform
        y, _ = librosa.load(wav_file, sr=sr)

        # Window và hop size
        win_length = int(0.025 * sr)   # 25ms = 400 samples
        hop_length = int(0.010 * sr)   # 10ms = 160 samples

        # STFT magnitude
        stft = librosa.stft(y, n_fft=512, win_length=win_length, hop_length=hop_length, window='hamming')
        mag = np.abs(stft[:64, :])  # Lấy 64 bins đầu tiên (low frequencies)

        # Log magnitude
        log_mag = np.log1p(mag)  # log(1 + x)

        # Transpose: (64, T) -> (T, 64)
        log_mag = log_mag.T

        # Frame stacking: 3 frames, skip = 3
        stacked_feats = []
        for i in range(0, len(log_mag) - 6, 3):  # skip rate = 3
            stacked = np.concatenate([log_mag[i], log_mag[i+3], log_mag[i+6]])
            stacked_feats.append(stacked)

        stacked_feats = torch.tensor(np.array(stacked_feats), dtype=torch.float)
        mean_feats = stacked_feats.mean(dim=0, keepdim=True)
        std_feats = stacked_feats.std(dim=0, keepdim=True)

        # stacked_feats = (stacked_feats - self.gmvn_mean) / (self.gmvn_std + 1e-5)
        if self.gmvn_mean is not None and self.gmvn_std is not None:
            stacked_feats = (stacked_feats - self.gmvn_mean) / (self.gmvn_std + 1e-5)
        else:
            stacked_feats = (stacked_feats - mean_feats) / (std_feats + 1e-5)
        return stacked_feats    

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        fbank = self.extract_features(wav_path)  # [T, 80]
        
        return {
            "text": encoded_text,        # [T_text]
            "fbank": fbank,              # [T_audio, 80]
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,  # [T_text + 1]
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def speech_collate_fn(batch):
    decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    text_mask=calculate_mask(text_lens, padded_texts.size(1))

    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask
    }

# ==============================================================

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# import torchaudio
# import torchaudio.transforms as T
# from tqdm import tqdm

# # [{idx : {encoded_text : Tensor, wav_path : text} }]


# def load_json(path):
#     """
#     Load a json file and return the content as a dictionary.
#     """
#     import json

#     with open(path, "r", encoding='utf-8') as f:
#         data = json.load(f)
#     return data

# class Vocab:
#     def __init__(self, vocab_path):
#         self.vocab = load_json(vocab_path)
#         self.itos = {v: k for k, v in self.vocab.items()}
#         self.stoi = self.vocab

#     def get_sos_token(self):
#         return self.stoi["<s>"]
#     def get_eos_token(self):
#         return self.stoi["</s>"]
#     def get_pad_token(self):
#         return self.stoi["<pad>"]
#     def get_unk_token(self):
#         return self.stoi["<unk>"]
#     def __len__(self):
#         return len(self.vocab)

# def compute_gmvn(dataset, sample_rate=16000):
#     mel_extractor = T.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_fft=512,
#         win_length=int(0.032 * sample_rate),
#         hop_length=int(0.010 * sample_rate),
#         n_mels=192,
#         power=2.0
#     )
#     sum_feats = torch.zeros(192)
#     sum_squares = torch.zeros(192)
#     total_frames = 0

#     for waveform in tqdm(dataset):  # dataset là list/tập của waveform tensors
#         with torch.no_grad():
#             mel = mel_extractor(waveform.unsqueeze(0))  # [1, 80, T]
#             log_mel = torchaudio.functional.amplitude_to_DB(
#                 mel, multiplier=10.0, amin=1e-10, db_multiplier=0
#             ).squeeze(0)  # [80, T]

#             total_frames += log_mel.shape[1]
#             sum_feats += log_mel.sum(dim=1)
#             sum_squares += (log_mel ** 2).sum(dim=1)

#     mean = sum_feats / total_frames
#     std = (sum_squares / total_frames - mean**2).sqrt()
#     return mean, std

# class Speech2Text(Dataset):
#     def __init__(self, json_path, vocab_path, cmvn_stats = None):
#         super().__init__()
#         self.data = load_json(json_path)
#         self.vocab = Vocab(vocab_path)
#         self.sos_token = self.vocab.get_sos_token()
#         self.eos_token = self.vocab.get_eos_token()
#         self.pad_token = self.vocab.get_pad_token()
#         self.unk_token = self.vocab.get_unk_token()
        
#         # stats = torch.load(cmvn_stats) 
#         # self.cmvn_mean = stats['mean']
#         # self.cmvn_std = stats['std']
            
#     def __len__(self):
#         return len(self.data)
    
#     # def get_fbank(self, waveform, sample_rate=16000):
#     #     mel_extractor = T.MelSpectrogram(
#     #         sample_rate=sample_rate,
#     #         n_fft=512,
#     #         win_length=int(0.032 * sample_rate),
#     #         hop_length=int(0.010 * sample_rate),
#     #         n_mels=80,  # ✨ để đúng với Conv1d(in_channels=80)
#     #         power=2.0
#     #     )

#     #     log_mel = mel_extractor(waveform.unsqueeze(0))
#     #     log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
#     #     log_mel = log_mel.squeeze(0)

#     #     # return log_mel.squeeze(0).transpose(0, 1)  # [T, 80]
        
#     #     # mean = log_mel.mean(dim=1, keepdim=True)
#     #     # std = log_mel.std(dim=1, keepdim=True)
#     #     # normalized_log_mel_spec = (log_mel - mean) / (std + 1e-5)

#     #     # spec = self.freq_mask(normalized_log_mel_spec)
#     #     # spec = self.time_mask(spec)

#     #     # return spec.transpose(0, 1)  # [T, 80]
    
#         # mean = log_mel.mean(dim=1, keepdim=True)
#         # std = log_mel.std(dim=1, keepdim=True)
#         # normalized_log_mel_spec = (log_mel - mean) / (std + 1e-5)

#     #     return normalized_log_mel_spec.transpose(0, 1)  # [T, 80]

#     def get_fbank(self, waveform, sample_rate=16000):
#         mel_extractor = T.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=512,
#             win_length=int(0.032 * sample_rate),
#             hop_length=int(0.01 * sample_rate),
#             n_mels=160,  # ✨ để đúng với Conv1d(in_channels=80)
#             power=2.0
#         )

#         log_mel = mel_extractor(waveform.unsqueeze(0))
#         log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        
#         features = log_mel.squeeze(0).transpose(0, 1) 

#         # features = (features - self.cmvn_mean) / (self.cmvn_std + 1e-5)
#         return features  # [T, 80]
    
#     # def get_fbank(self, waveform, sample_rate=16000):
#     #     mel_extractor = T.MelSpectrogram(
#     #         sample_rate=sample_rate,
#     #         n_fft=512,
#     #         win_length=int(0.025 * sample_rate),  # 25ms
#     #         hop_length=int(0.010 * sample_rate),  # 10ms
#     #         n_mels=64,  # để còn stack thành 192
#     #         power=2.0
#     #     )

#     #     log_mel = mel_extractor(waveform.unsqueeze(0))  # [1, 64, T]
#     #     log_mel = torchaudio.functional.amplitude_to_DB(
#     #         log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0
#     #     )  # [1, 64, T]
        
#     #     # Frame stacking 3 frames với skip=3
#     #     def stack_frames(feat, stack=3, skip=3):
#     #         B, D, T = feat.shape
#     #         T_new = T - (stack - 1) * skip
#     #         out = []
#     #         for i in range(stack):
#     #             out.append(feat[:, :, i*skip : i*skip + T_new])
#     #         return torch.cat(out, dim=1)  # [B, 64*3, T_new]

#     #     stacked = stack_frames(log_mel, stack=3, skip=3)  # [1, 192, T']
#     #     features = stacked.squeeze(0).transpose(0, 1)  # [T', 192]

#     #     # Step 3: Apply global CMVN
#     #     features = (features - self.cmvn_mean) / (self.cmvn_std + 1e-5)

#     #     return features
    
#     def extract_from_path(self, wave_path):
#         waveform, sr = torchaudio.load(wave_path)
#         waveform = waveform.squeeze(0)  # (channel,) -> (samples,)
#         return self.get_fbank(waveform, sample_rate=sr)

#     def __getitem__(self, idx):
#         current_item = self.data[idx]
#         wav_path = current_item["wav_path"]
#         encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
#         decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
#         fbank = self.extract_from_path(wav_path).float()  # [T, 80]
        
#         return {
#             "text": encoded_text,        # [T_text]
#             "fbank": fbank,              # [T_audio, 80]
#             "text_len": len(encoded_text),
#             "fbank_len": fbank.shape[0],
#             "decoder_input": decoder_input,  # [T_text + 1]
#         }
    
# from torch.nn.utils.rnn import pad_sequence

# def calculate_mask(lengths, max_len):
#     """Tạo mask cho các tensor có chiều dài khác nhau"""
#     mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
#     return mask

# def speech_collate_fn(batch):
#     decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
#     texts = [item["text"] for item in batch]
#     fbanks = [item["fbank"] for item in batch]
#     text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
#     fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

#     padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
#     padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
#     padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

#     speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
#     text_mask=calculate_mask(text_lens, padded_texts.size(1))

#     return {
#         "decoder_input": padded_decoder_inputs,
#         "text": padded_texts,
#         "text_mask": text_mask,
#         "text_len" : text_lens,
#         "fbank_len" : fbank_lens,
#         "fbank": padded_fbanks,
#         "fbank_mask": speech_mask
#     }