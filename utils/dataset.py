import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
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

def compute_gmvn(dataset, sample_rate=16000):
    wav_files = glob(os.path.join(dataset, "**", "*.wav"), recursive=True)
    mel_extractor = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=int(0.032 * sample_rate),
        hop_length=int(0.010 * sample_rate),
        n_mels=192,
        power=2.0
    )
    sum_feats = torch.zeros(192)
    sum_squares = torch.zeros(192)
    total_frames = 0

    for file in tqdm(wav_files):  # dataset là list/tập của waveform tensors
        with torch.no_grad():
            waveform, sr = torchaudio.load(file)
            mel = mel_extractor(waveform)  # [1, 192, T]
            log_mel = torchaudio.functional.amplitude_to_DB(
                mel, multiplier=10.0, amin=1e-10, db_multiplier=0
            ).squeeze(0)  # [192, T]

            total_frames += log_mel.shape[1]
            sum_feats += log_mel.sum(dim=1)
            sum_squares += (log_mel ** 2).sum(dim=1)

    mean = sum_feats / total_frames
    std = (sum_squares / total_frames - mean ** 2).sqrt()

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
            
    def __len__(self):
        return len(self.data)

    def get_fbank(self, waveform, sample_rate=16000):
        mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(0.032 * sample_rate),
            hop_length=int(0.01 * sample_rate),
            n_mels=192,  
            power=2.0
        )

        log_mel = mel_extractor(waveform.unsqueeze(0))
        log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        log_mel = log_mel.squeeze(0)
        print(f"Log mel shape: {log_mel.shape}")  # Debugging line
        features = (log_mel - self.gmvn_mean.unsqueeze(1)) / (self.gmvn_std.unsqueeze(1) + 1e-5)
        features = log_mel.transpose(0, 1)  # [T, 192]
        return features  # [T, 80]
    
    def extract_from_path(self, wave_path):
        waveform, sr = torchaudio.load(wave_path)
        waveform = waveform.squeeze(0)  # (channel,) -> (samples,)
        return self.get_fbank(waveform, sample_rate=sr)

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        fbank = self.extract_from_path(wav_path).float()  # [T, 80]
        
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





# def get_fbank(self, waveform, sample_rate=16000):
#     mel_extractor = T.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_fft=512,
#         win_length=int(0.032 * sample_rate),
#         hop_length=int(0.010 * sample_rate),
#         n_mels=80,
#         power=2.0
#     )

#     log_mel = mel_extractor(waveform.unsqueeze(0))
#     log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
#     log_mel = log_mel.squeeze(0)  # [80, T]

#     # 🔥 Áp dụng global CMVN
#     log_mel = (log_mel - self.cmvn_mean.unsqueeze(1)) / (self.cmvn_std.unsqueeze(1) + 1e-5)

#     return log_mel.transpose(0, 1)  # [T, 80]

# Lưu
# torch.save({'mean': mean, 'std': std}, 'cmvn_stats.pt')

# # Load
# stats = torch.load('cmvn_stats.pt')
# self.cmvn_mean = stats['mean']
# self.cmvn_std = stats['std']
