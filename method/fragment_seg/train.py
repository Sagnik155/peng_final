import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.split import get_train_val_splits
from data.dataset import PengwinFragmentDataset
from fragment_seg.unet3d import LightweightFragmentUNet
from fragment_seg.losses import FragmentLoss

def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print (f"Using device: {device}")

    train_cases, val_cases = get_train_val_splits(val_ratio=0.2)

    train_dataset = PengwinFragmentDataset(train_cases, click_strategy="uniformly_sampled", is_train=True)

    is_cuda = (device.type == "cuda")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4 if device.type != "cpu" else 0,
        pin_memory=is_cuda
    )

    model = LightweightFragmentUNet(in_channels=2, out_channels=2, base_filters=16).to(device)
    criterion = FragmentLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    for epoch in range (epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch["input"].to(device, non_blocking=is_cuda)
            labels = batch["label"].to(device, non_blocking=is_cuda)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    save_path = os.path.join(os.path.dirname(__file__), "fragment_unet_test.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    train_model()
