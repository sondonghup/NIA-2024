import numpy as np
import logging
import torch
import os

from tqdm import tqdm
from torch.cuda.amp import autocast

from utils import round_to_nearest_valid_score, top_k_score

def test_epoch(model, valid_iter, device):

    model.eval()

    all_preds = list()
    all_labels = list()

    with torch.no_grad():
        
        for batch in tqdm(valid_iter, desc='test'):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
            
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        conv_preds = np.array([round_to_nearest_valid_score(score) for score in all_preds.flatten()])
        conv_labels = np.array([round_to_nearest_valid_score(score) for score in all_labels.flatten()])

        for k in [1, 2, 3]:
            top_k_acc = top_k_score(conv_labels, conv_preds, k)
            logging.info(f' Top-{k} f1-score: {top_k_acc:.4f}')

    return [list(conv_preds[i:i + 8]) for i in range(0, len(conv_preds), 8)]