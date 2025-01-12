# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import ChessDataset
from model import ChessNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import math  # For tanh
from torch.cuda.amp import autocast, GradScaler

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def plot_losses(epochs_range, train_policy_losses, train_value_losses, val_policy_losses, val_value_losses, save_path):
    plt.figure(figsize=(12, 5))

    # Policy Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_policy_losses, label='Train Policy Loss')
    plt.plot(epochs_range, val_policy_losses, label='Val Policy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss Over Epochs')
    plt.legend()

    # Value Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_value_losses, label='Train Value Loss')
    plt.plot(epochs_range, val_value_losses, label='Val Value Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value Loss')
    plt.title('Value Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_checkpoint(resume_from, model, optimizer, scheduler):
    checkpoint = torch.load(resume_from, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed training from checkpoint: {resume_from}, starting at epoch {start_epoch}")
    else:
        # Assume checkpoint is the state_dict
        model.load_state_dict(checkpoint)
        print(f"Loaded model state_dict from checkpoint: {resume_from}")
        start_epoch = 1
        best_val_loss = float('inf')  # Reset best_val_loss if not present

    return start_epoch, best_val_loss

def train(resume_from=None):
    h5_folder_path = 'pgn'
    batch_size = 256
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4 
    num_workers = 8
    scaling_factor = 500  
    max_cp = 1500

    pth_dir = 'pth'
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h5_file_paths = [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.endswith('.h5')]

    if not h5_file_paths:
        print(f"No .h5 files found in {h5_folder_path} directory.")
        return

    # Dataset
    datasets = [ChessDataset(h5_file) for h5_file in h5_file_paths]
    combined_dataset = ConcatDataset(datasets)

    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = ChessNet().to(device)
    model.apply(weights_init_he)

    # Optimizer and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Lists to store losses for plotting
    train_policy_losses = []
    train_value_losses = []
    val_policy_losses = []
    val_value_losses = []

    # Gradscalar for mixed precision
    scaler = GradScaler()

    start_epoch = 1
    best_val_loss = float('inf')
    if resume_from:
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer, scheduler)

    # Training Loop
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            running_policy_loss = 0.0
            running_value_loss = 0.0

            # Progress bar setup
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch')

            for inputs, policy_targets, value_targets in progress_bar:
                inputs = inputs.to(device)  # (batch_size, 17, 8, 8)
                policy_targets = policy_targets.to(device).long()
                value_targets = value_targets.to(device).float()

                score = value_targets * max_cp

                # Apply the new scaling using tanh
                value_targets = torch.tanh(score / scaling_factor)

                optimizer.zero_grad()

                with autocast():
                    policy_outputs, value_outputs = model(inputs)  # Policy output is 4084

                    # Policy loss
                    policy_loss = policy_criterion(policy_outputs, policy_targets)

                    # Value loss
                    value_outputs = value_outputs.squeeze()
                    value_loss = value_criterion(value_outputs, value_targets)

                    # Total loss
                    total_loss = policy_loss + value_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_policy_loss += policy_loss.item() * inputs.size(0)
                running_value_loss += value_loss.item() * inputs.size(0)

                progress_bar.set_postfix({'Policy Loss': f"{policy_loss.item():.4f}", 'Value Loss': f"{value_loss.item():.4f}"})

            epoch_policy_loss = running_policy_loss / len(train_dataset)
            epoch_value_loss = running_value_loss / len(train_dataset)
            train_policy_losses.append(epoch_policy_loss)
            train_value_losses.append(epoch_value_loss)

            model.eval()
            val_running_policy_loss = 0.0
            val_running_value_loss = 0.0
            with torch.no_grad():
                for inputs, policy_targets, value_targets in val_loader:
                    inputs = inputs.to(device)
                    policy_targets = policy_targets.to(device).long()
                    value_targets = value_targets.to(device).float()

                    # Reverse engineer the original score from value_targets
                    score = value_targets * max_cp

                    # Apply the new scaling using tanh
                    value_targets = torch.tanh(score / scaling_factor)

                    policy_outputs, value_outputs = model(inputs)

                    # Policy loss
                    policy_loss = policy_criterion(policy_outputs, policy_targets)

                    # Value loss
                    value_outputs = value_outputs.squeeze()
                    value_loss = value_criterion(value_outputs, value_targets)

                    val_running_policy_loss += policy_loss.item() * inputs.size(0)
                    val_running_value_loss += value_loss.item() * inputs.size(0)

            val_policy_loss = val_running_policy_loss / len(val_dataset)
            val_value_loss = val_running_value_loss / len(val_dataset)
            val_total_loss = val_policy_loss + val_value_loss

            val_policy_losses.append(val_policy_loss)
            val_value_losses.append(val_value_loss)

            print(f'Epoch {epoch}/{num_epochs}, '
                  f'Train Policy Loss: {epoch_policy_loss:.4f}, Train Value Loss: {epoch_value_loss:.4f}, '
                  f'Val Policy Loss: {val_policy_loss:.4f}, Val Value Loss: {val_value_loss:.4f}')

            # Scheduler step based on validation loss
            scheduler.step(val_total_loss)

            # Save the best model based on validation loss
            if val_total_loss < best_val_loss - 1e-3:
                best_val_loss = val_total_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_total_loss
                }, 'best_model.pth')
                print("Best model saved.")

            # Save model every 3 epochs
            if epoch % 3 == 0:
                checkpoint_path = os.path.join(pth_dir, f'epoch{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_total_loss
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            epochs_range = range(1, epoch + 1)
            save_path = f'loss_over_epochs_epoch{epoch}.png'
            plot_losses(epochs_range, train_policy_losses, train_value_losses, val_policy_losses, val_value_losses, save_path)
            print(f"Saved loss plot: {save_path}")

    except KeyboardInterrupt:
        print("\nSaving current loss plots, keyboard interrupt")
        current_epoch = epoch if 'epoch' in locals() else 0
        if current_epoch > 0:
            epochs_range = range(1, current_epoch + 1)
            save_path = f'loss_over_epochs_epoch{current_epoch}.png'
            plot_losses(epochs_range, train_policy_losses, train_value_losses, val_policy_losses, val_value_losses, save_path)
            print(f"Saved loss plot up to epoch {current_epoch}: {save_path}")
        else:
            print("Nothing to save")
        print("Exiting training")

    if 'epoch' in locals() and epoch == num_epochs:
        epochs_range = range(1, epoch + 1)
        save_path = 'loss_over_epochs_final.png'
        plot_losses(epochs_range, train_policy_losses, train_value_losses, val_policy_losses, val_value_losses, save_path)
        print(f"Saved final loss plot: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the chessnet model')
    parser.add_argument('--resume', type=str, default=None, help='path to the .pth to resume training')

    args = parser.parse_args()

    train(resume_from=args.resume)
