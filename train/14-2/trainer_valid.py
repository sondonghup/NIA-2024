import numpy as np
import logging
import torch
import os

from tqdm import tqdm
from torch.cuda.amp import autocast

from utils import round_to_nearest_valid_score, top_k_score

def valid_epoch(model,
                valid_iter,
                loss_fn,
                device,
                save_dir,
                compare_valid_loss = float('inf') ):

    model.eval()

    losses = list()
    all_preds = list()
    all_labels = list()

    with torch.no_grad():
        
        for batch in tqdm(valid_iter, desc='valid'):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
            
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = loss_fn(outputs, labels)

            losses.append(loss.item())

            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        conv_preds = np.array([round_to_nearest_valid_score(score) for score in all_preds.flatten()])
        conv_labels = np.array([round_to_nearest_valid_score(score) for score in all_labels.flatten()])
        
        loss = np.mean(losses)
        logging.info(f" Average valid loss : {loss:.4f}")

        for k in [1, 2, 3]:
            top_k_acc = top_k_score(conv_labels, conv_preds, k)
            logging.info(f' Top-{k} f1-score: {top_k_acc:.4f}')

        if compare_valid_loss > loss:

            if not os.path.exists(f"{save_dir}"):
                os.makedirs(f"{save_dir}")

            if not os.path.exists(f"{save_dir}valid_loss_{loss}/"):
                os.makedirs(f"{save_dir}valid_loss_{loss}/")

            model.class_model.save_pretrained(f"{save_dir}valid_loss_{loss}/")

            torch.save({
                'fc_state_dict': model.fc.state_dict(),
                'dropout': model.drop.p,
            }, os.path.join(f"{save_dir}valid_loss_{loss}/", 'other_weights.pt'))
            compare_valid_loss = loss

    return conv_preds