import os
import torch
import torch.nn as nn
import json
import argparse
import logging

from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader

from utils import get_file_list
from data_load import make_dataset
from trainer_test import test_epoch


def load_model(model_name, save_dir, device, token):
    
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory not found: {save_dir}")
    
    model_folders = [f for f in os.listdir(save_dir) if f.startswith('valid_loss_')]
    if not model_folders:
        raise ValueError(f"No valid_loss folders found in {save_dir}")
    
    best_folder = min(model_folders, key=lambda x: float(x.split('_')[-1]))
    model_path = os.path.join(save_dir, best_folder)
    
    logging.info(f"Loading model from: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    logging.info("Loading base model...")
    base_model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        token=token
    )

    class LoadedClassifier(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.class_model = base_model
            self.drop = nn.Dropout(p=0.3)
            self.fc = nn.Linear(self.class_model.config.hidden_size, 8)

        def forward(self, input_ids, attention_mask):
            pooled_output = self.class_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled_output = self.drop(pooled_output.last_hidden_state.mean(dim=1))
            scores = self.fc(pooled_output)
            return scores

    logging.info("Creating model instance...")
    model = LoadedClassifier(base_model)

    logging.info("Loading LoRA weights...")
    model.class_model = PeftModel.from_pretrained(
        model.class_model,
        model_path,
        is_trainable=False
    )

    logging.info("Loading other weights...")
    other_weights_path = os.path.join(model_path, 'other_weights.pt')
    if os.path.exists(other_weights_path):
        other_weights = torch.load(other_weights_path, map_location=device)
        model.fc.load_state_dict(other_weights['fc_state_dict'])
        model.drop.p = other_weights['dropout']
    
    model.to(device)
    
    logging.info("Model loading completed!")
    return model

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', dest = 'base_model_name', type = str)
    parser.add_argument('--trained_model_path', dest = 'trained_model_path', type = str)
    parser.add_argument('--device', dest = 'device', type = str)
    parser.add_argument('--test_data_path', dest = 'test_data_path', type = str)
    parser.add_argument('--token', dest = 'token', type = str)
    parser.add_argument('--num_features', dest = 'num_features', type = int)
    parser.add_argument('--batch_size', dest = 'batch_size', type = int)
    parser.add_argument('--result_name', dest = 'result_name', type = str)
    parser.add_argument('--log_name', dest = 'log_name', type = str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
    logging.FileHandler(args.log_name),
    logging.StreamHandler()
])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, token = args.token)
    tokenizer.pad_token = tokenizer.eos_token
    test_file_list = get_file_list(args.test_data_path)
    model = load_model(args.base_model_name, args.trained_model_path, args.device, token = args.token)
    
    test_data = make_dataset(test_file_list, tokenizer, args.num_features)
    test_dataloader = DataLoader(test_data,
                                  batch_size = args.batch_size,
                                  shuffle = False,)

    preds = test_epoch(model,
               test_dataloader,
               args.device
               )

    data_list = list()

    for pred, data in tqdm(zip(preds, test_dataloader), desc = 'gathering...'):
        data_dict = dict()
        data_dict['text'] = data['text'][0]
        data_dict['pred'] = pred
        data_list.append(data_dict)

    with open(args.result_name, 'w', encoding = 'utf-8')as f:
        json.dump(data_list, f, ensure_ascii = False, indent = 4)