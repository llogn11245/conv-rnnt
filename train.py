import torch
from utils.dataset import Speech2Text, speech_collate_fn, compute_gmvn
from models.model import Transducer
from tqdm import tqdm
from models.loss import RNNTLoss
import argparse
import yaml
import os 
from models.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

# Cấu hình logger
log_file = "conv_rnnt_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # vẫn in ra màn hình
    ]
)

def reload_model(model, optimizer, checkpoint_path, model_name):
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    print(path_list)
    if len(path_list) > 0:
        for path in path_list:
            try:
                past_epoch = max(int(path.split("_")[-1]), past_epoch)
            except:
                continue
        
        load_path = os.path.join(checkpoint_path, f"{model_name}_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Reloaded model from {load_path} at epoch {past_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")
    
    return past_epoch + 1, model, optimizer


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch in progress_bar:
        speech = batch["fbank"].to(device)
        speech_mask = batch["fbank_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        fbank_len = batch["fbank_len"].to(device)
        text_len = batch["text_len"].to(device)
        target_text = batch["text"].to(device)
        decoder_input = batch["decoder_input"].to(device)

        optimizer.zero_grad()
        output = model(speech, fbank_len.long(), decoder_input.int(), text_len.long())
        loss = criterion(output, target_text, fbank_len, text_len)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            speech = batch["fbank"].to(device)
            target_text = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            decoder_input = batch["decoder_input"].to(device)

            output = model(speech, fbank_len.long(), decoder_input.int(), text_len.long())
            loss = criterion(output, target_text, fbank_len, text_len)

            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average validation loss: {avg_loss:.4f}")
    return avg_loss


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']

    # ==== Compute gmvn ====
    gmvn_mean, gmvn_std = compute_gmvn(training_cfg['voice_path'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
        gmvn_mean = gmvn_mean,
        gmvn_std = gmvn_std,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn
    )

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path'],
        gmvn_mean = gmvn_mean,
        gmvn_std = gmvn_std,
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn
    )

    # ==== Model ====
    model = Transducer(config['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==== Loss ====
    criterion = RNNTLoss(config["rnnt_loss"]["blank"], config["rnnt_loss"]["reduction"])

    # ==== Optimizer ====
    optimizer = Optimizer(model.parameters(), config['optim'])

    # ==== Scheduler ====
    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,  # because you're using a wrapper class
        mode='min',
        factor=0.5,
        patience=2,
        # verbose=True
    )

    # ==== Reload checkpoint if needed ====
    start_epoch = 1
    if training_cfg['reload']:
        checkpoint_path = training_cfg['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path, config['model']['name'])

    # ==== Training loop ====
    num_epochs = training_cfg["epochs"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, dev_loader, criterion, device)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save model checkpoint
        model_filename = os.path.join(
            training_cfg['save_path'],
            f"{config['model']['name']}_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        # Step scheduler with validation loss
        scheduler.step(val_loss)

        # Early stopping nếu lr quá nhỏ
        current_lr = optimizer.optimizer.param_groups[0]["lr"]
        if current_lr < 1e-6:
            logging.info('Learning rate quá thấp. Kết thúc training.')
            break


if __name__ == "__main__":
    main()
