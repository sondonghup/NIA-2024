import numpy as np
import logging
import torch

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils import round_to_nearest_valid_score, top_k_score

def train_epoch(model, 
                train_iter, 
                loss_fn, 
                optimizer, 
                device, 
                scheduler):
    
    model.train()

    losses = list()
    all_preds = list()
    all_labels = list()
    
    scaler = GradScaler()

    for batch in tqdm(train_iter, desc='train'):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        losses.append(loss.item())

        all_preds.extend(outputs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conv_preds = np.array([round_to_nearest_valid_score(score) for score in all_preds.flatten()])
    conv_labels = np.array([round_to_nearest_valid_score(score) for score in all_labels.flatten()])

    loss = np.mean(losses)
    logging.info(f" Average train loss : {loss:.4f}")

    for k in [1, 2, 3]:
        top_k_acc = top_k_score(conv_labels, conv_preds, k)
        logging.info(f" Top-{k} f1-score : {top_k_acc:.4f}") 

    logging.info(f"train 종료 gpu 사용량 : {torch.cuda.memory_allocated() / 1024 /1024}")

    return conv_preds