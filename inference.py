import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import Transducer
from utils.dataset import Speech2Text, speech_collate_fn
from jiwer import wer, cer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ids_to_text(ids, itos, eos_id=None):
    tokens = []
    for idx in ids:
        if eos_id is not None and idx == eos_id:
            break
        token = itos.get(idx, '')
        if token in ['<pad>','<s>','</s>','<unk>','<blank>']:
            continue
        tokens.append(token)
    return ' '.join(tokens)

def main():
    parser = argparse.ArgumentParser(description="Inference script for RNN-T speech-to-text model")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_json', required=True, help='Path to test JSON data')
    parser.add_argument('--vocab_json', required=True, help='Path to vocab JSON file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output', default='results.csv', help='CSV file to save predictions')
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    model_cfg = full_cfg.get('model', full_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===Load Checkpoint===
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    #===Load Model===
    model = Transducer(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #===Load Data===
    dataset = Speech2Text(args.test_json, args.vocab_json, args.cmvn_stats)
    itos    = dataset.vocab.itos
    eos_id  = dataset.vocab.get_eos_token()

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=speech_collate_fn)

    pred_texts = []
    true_texts = []

    with open(args.output, 'w', encoding='utf-8') as fout:
        for batch in loader:
            fbanks     = batch['fbank'].to(device)
            fbank_lens = batch['fbank_len'].to(device)

            with torch.no_grad():
                batch_preds = model.recognize(fbanks, fbank_lens)

            for i in range(len(batch_preds)):
                pred_ids = batch_preds[i]
                true_ids = batch['text'][i].tolist()

                pred_text = ids_to_text(pred_ids, itos, eos_id=eos_id)
                true_text = ids_to_text(true_ids, itos, eos_id=eos_id)

                pred_texts.append(pred_text)
                true_texts.append(true_text)
                print(f"Predict text: {pred_text}")
                print(f"Ground truth: {true_text}")
                fout.write(f"Predict text: {pred_text}\n")
                fout.write(f"Ground truth: {true_text}\n")
                fout.write("---------------\n")

    print(f"Inference complete. Results saved to {args.output}")

    #===TÍNH WER VÀ CER===
    overall_wer = wer(true_texts, pred_texts)
    overall_cer = cer(true_texts, pred_texts)
    print(f"Word Error Rate (WER): {overall_wer:.4f}")
    print(f"Character Error Rate (CER): {overall_cer:.4f}")

if __name__ == '__main__':
    main()

# python /data/npl/Speech2Text/rna/conv-rnnt/inference.py \
#     --config /data/npl/Speech2Text/rna/conv-rnnt/configs/conv_rnnt.yaml \
#     --checkpoint /data/npl/Speech2Text/rna/conv-rnnt/0506_rnnt_datset_class_2/conv-rnnt_epoch_4 \
#     --test_json /data/npl/Speech2Text/rna/transformer_transducer/data/test_w2i.json \
#     --vocab_json /data/npl/Speech2Text/rna/transformer_transducer/data/vocab_w2i.json \
#     --cmvn_stats /data/npl/Speech2Text/rna/zlinhtinh/cmvn_stats.pt \
#     --batch_size 1 \
#     --output /data/npl/Speech2Text/rna/conv-rnnt/predictions.txt 