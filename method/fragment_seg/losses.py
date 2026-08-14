import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors for calculation
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FragmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        # We use Channel 0 for the main prediction.
        # Channel 1 is reserved for the upcoming 20mm physical boundary loss.
        pred_logits = logits[:, 0:1, ...]
        
        bce_loss = self.bce(pred_logits, targets)
        dice_loss = self.dice(pred_logits, targets)
        
        return bce_loss + dice_loss