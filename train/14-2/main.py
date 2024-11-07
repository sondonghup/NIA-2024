import argparse
import logging
import torch
import os
import torch.nn as nn
import bitsandbytes as bnb

from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_load import make_dataset
from utils import get_file_list
from model import Classifier
from trainer_train import train_epoch
from trainer_valid import valid_epoch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', dest = 'log_path', type = str)
    parser.add_argument('--batch_size', dest = 'batch_size', type = int)
    parser.add_argument('--learning_rate', dest = 'learning_rate', type = float)
    parser.add_argument('--epoch', dest = 'epoch', type = int)
    parser.add_argument('--train_file_path', dest = 'train_file_path', type = str)
    parser.add_argument('--valid_file_path', dest = 'valid_file_path', type = str)
    parser.add_argument('--test_file_path', dest = 'test_file_path', type = str)
    parser.add_argument('--save_path', dest = 'save_path', type = str)
    parser.add_argument('--model_name', dest = 'model_name', type = str)
    parser.add_argument('--hf_token', dest = 'hf_token', type = str)
    parser.add_argument('--num_feature', dest = 'num_feature', type = int)
    parser.add_argument('--load_in_4bit', dest = 'load_in_4bit', type = bool)
    parser.add_argument('--bnb_4bit_quant_type', dest = 'bnb_4bit_quant_type', type = str)
    parser.add_argument('--bnb_4bit_use_double_quant', dest = 'bnb_4bit_use_double_quant', type = bool)
    parser.add_argument('--lora_rank', dest = 'lora_rank', type = int)
    parser.add_argument('--lora_alpha', dest = 'lora_alpha', type = int)
    parser.add_argument('--target_modules', dest = 'target_modules', type = list)
    parser.add_argument('--lora_dropout', dest = 'lora_dropout', type = float)
    parser.add_argument('--bias', dest = 'bias', type = str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_path),
                                  logging.StreamHandler()
                         ])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token = args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    train_file_list = get_file_list(args.train_file_path)
    valid_file_list = get_file_list(args.valid_file_path)
    test_file_list = get_file_list(args.test_file_path)

    train_data = make_dataset(train_file_list, tokenizer, args.num_feature)
    train_dataloader = DataLoader(train_data,
                                  batch_size = args.batch_size,
                                  shuffle = True,)

    valid_data = make_dataset(valid_file_list, tokenizer, args.num_feature) 
    valid_dataloader = DataLoader(valid_data,
                                  batch_size = args.batch_size,
                                  shuffle = False,)

    model = Classifier(model_name = args.model_name,
                       token = args.hf_token
                       load_in_4bit = args.load_in_4bit,
                       bnb_4bit_quant_type = args.bnb_4bit_quant_type,
                       bnb_4bit_use_double_quant = args.bnb_4bit_use_double_quant,
                       r = args.lora_rank,
                       lora_alpha = args.lora_alpha,
                       target_modules = args.target_modules,
                       lora_dropout = args.lora_dropout,
                       bias = args.bias
                       )

    model = model.to(device)

    save_dir = args.save_path

    optimizer = AdamW(model.parameters(), lr = args.learning_rate)

    total_steps = len(train_dataloader) * 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 1e-6)

    loss_fn = nn.MSELoss()

    for epoch in range(args.epoch):
        logging.info(f'Epoch {epoch + 1}/{args.epoch}')
        logging.info(f'-' * 10)

        _  = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            scheduler,
        )

    
        _ = valid_epoch(
            model,
            valid_dataloader,
            loss_fn,
            device,
            save_dir,
        )