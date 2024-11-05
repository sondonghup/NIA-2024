import json
import torch
import numpy as np
import random

from torch.utils.data import Dataset

class make_dataset(Dataset):
    def __init__(self, files, tokenizer, num_feautre):

        self._files = files
        self._tokenizer = tokenizer
        self._num_feature = num_feautre

        random.seed(42)

    def __len__(self):

        return len(self._files)

    def __getitem__(self, index):

        def _make_int(data):

            return (sum(data) / 2)

        file = self._files[index]
        label = list()
        
        if file.endswith(".json"):
            with open(file, encoding='utf-8') as f:
                row = json.load(f)

            question = row['essay_question']['prompt']
            paragraph_text = row['essay_answer']['text']
            rubric = row['rubric']['analytic']

            del row['rubric']['analytic']['task_1']['rubric_key']
            del row['rubric']['analytic']['content_1']['rubric_key']
            del row['rubric']['analytic']['content_2']['rubric_key']
            del row['rubric']['analytic']['content_3']['rubric_key']
            del row['rubric']['analytic']['organization_1']['rubric_key']
            del row['rubric']['analytic']['organization_2']['rubric_key']
            del row['rubric']['analytic']['expression_1']['rubric_key']
            del row['rubric']['analytic']['expression_2']['rubric_key']
            
            if 'UNKNOWN' in row.get('essay_answer', {}).get('feature', {}):
                del row['essay_answer']['feature']['UNKNOWN']

            keyword = row['essay_question']['keyword']

            grading_feature = ' '.join([f"{data} : {row['essay_answer']['feature'][data]}"  for data in random.sample(list(row['essay_answer']['feature'].keys()), self._num_feature)])

            input_text = f"질문 : {question}, 답변 : {paragraph_text}, 키워드 : {keyword}, 루브릭 : {rubric}, 채점자질 : {grading_feature}" # 입력

            encoded_input_text = self._tokenizer.encode_plus(
                input_text,
                add_special_tokens = True,
                return_token_type_ids = False,
                truncation = True,
                padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
                max_length = 2000
            ) 

            scores = row["score"]["personal"]["analytic"]
            labels = [np.mean([int(s) for s in score["score"]]) for score in scores.values()]

            return {
                'text' : input_text,
                'input_ids' : encoded_input_text['input_ids'].squeeze(),
                'attention_mask' : encoded_input_text['attention_mask'].squeeze(),
                'labels' : torch.tensor(labels, dtype=torch.float),
            }