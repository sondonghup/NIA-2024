import logging
import torch.nn as nn
import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoModel, BitsAndBytesConfig

class Classifier(nn.Module):
    def __init__(self, 
                 model_name,
                 token,
                 load_in_4bit,
                 bnb_4bit_quant_type,
                 bnb_4bit_use_double_quant,
                 r,
                 lora_alpha,
                 target_modules,
                 lora_dropout,
                 bias,
                 ):

        super(Classifier, self).__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = load_in_4bit,
            bnb_4bit_quant_type = bnb_4bit_quant_type,
            bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        )

        self.class_model = AutoModel.from_pretrained(model_name,
                                                     quantization_config = bnb_config,
                                                     token = token)

        lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias
        )

        self.class_model = get_peft_model(self.class_model, lora_config)
        self.drop = nn.Dropout(p = 0.3)
        self.fc = nn.Linear(self.class_model.config.hidden_size, 8)

    def forward(self, input_ids, attention_mask):
        
        pooled_output = self.class_model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
        )

        pooled_output = self.drop(pooled_output.last_hidden_state.mean(dim=1))
        scores = self.fc(pooled_output)

        return scores