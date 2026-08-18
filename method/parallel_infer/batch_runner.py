import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from roi.roi_extractor import extract_inference_roi, place_roi_back

def run_inference_on_clicks(img_array: np.ndarray, clicks: list, model: torch.nn.Module, device: torch.device) -> list[dict]:
    model.eval()
    roi_predictions = []
    
    with torch.no_grad():
        for click in clicks:
            anatomy = click['anatomy']
            if anatomy == "background":
                continue
                
            x_tensor, placement_info = extract_inference_roi(img_array, click['z'], click['y'], click['x'])
            x_tensor = x_tensor.to(device)

            logits = model(x_tensor)

            probs = torch.sigmoid(logits[0, 0, ...])
            print(f"Max probability for {anatomy}: {probs.max().item():.4f}")

            pred_mask = probs.cpu().numpy() > 0.01

            global_mask = place_roi_back(pred_mask, placement_info)

            roi_predictions.append({
                'global_mask': global_mask,
                'anatomy': anatomy
            })
            
    return roi_predictions
