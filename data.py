import pickle

import torch
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):

    typed_entity_marker = ['[S:PER]', '[/S:PER]', '[S:ORG]', '[/S:ORG]',
                           '[O:PER]', '[/O:PER]', '[O:ORG]', '[/O:ORG]',
                           '[O:DAT]', '[/O:DAT]', '[O:LOC]', '[/O:LOC]',
                           '[O:POH]', '[/O:POH]', '[O:NOH]', '[/O:NOH]',]

    def __init__(self, model_name="klue/bert-base", batch_size=128, version=2):
        super().__init__()
        
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if version == 1:
            # ... sentence ...  ->  <sub_entity>[SEP]<ob_entity>[SEP]... sentence ...
            self.preprocess = self.data_preprocess1 
        else:
            # ... sentence ...  -> ... [S:type]sub_entity[/S:type] ... [O:type]ob_entity[/O:type] ...
            self.preprocess = self.data_preprocess2
            self.num_added_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": self.typed_entity_marker})
            # typed-entity-marker를 위한 special token 추가
            # model 에서 embedding layer 바꿔주어야 함.
        
        if "roberta" in model_name:
            # RoBERTa 모델을 사용할 경우 token_type_ids를 반환하지 않도록 설정
            self.return_token_type_ids = False
        else:
            self.return_token_type_ids = True

    def prepare_data(self):
        klue_re = load_dataset('klue', 're')
        self.train_data = klue_re['train']
        self.valid_data = klue_re['validation']
        
        with open('/opt/ml/klue/dict_klue_label_mapper.pkl', 'rb') as f:
            self.label_mapper = pickle.load(f) # klue data lavel -> train.csv data label
        
        self.train_data = self.train_data.map(self.preprocess, num_proc=4, remove_columns=["subject_entity", "object_entity"])
        self.valid_data = self.valid_data.map(self.preprocess, num_proc=4, remove_columns=["subject_entity", "object_entity"])

    def data_preprocess1(self, example):
        example["label"] = self.label_mapper[example["label"]]
        return {"entity_span": example["subject_entity"]["word"] + "[SEP]" + example["object_entity"]["word"]}
        
    def data_preprocess2(self, example):
        example["label"] = self.label_mapper[example["label"]]
        # sentence에 special token 추가된 문장으로 바꾸기.
        # 이 부분 더 효율적으로 개선할 방법 고민해보기.
        sentence = example["sentence"]
        sub_w, sub_s, sub_e, sub_t = example["subject_entity"].values()
        ob_w, ob_s, ob_e, ob_t = example["object_entity"].values()
        sentence = ''.join([sentence[:sub_s], f'[S:{sub_t}]', sub_w, f'[/S:{sub_t}]', sentence[sub_e+1:]])
        sentence = ''.join([sentence[:ob_s], f'[O:{ob_t}]', ob_w, f'[/O:{ob_t}]', sentence[ob_e+1:]])
        example["sentence"] = sentence
        return example
        
    def tokenize_data(self, example):
        if "entity_span" in example.keys():
            return self.tokenizer(
                example["entity_span"],
                example["sentence"], 
                truncation=True, 
                padding="max_length", 
                max_length=256, 
                add_special_tokens=True,
                return_token_type_ids=self.return_token_type_ids,
            )
        else:
            return self.tokenizer(
                example["sentence"], 
                truncation=True, 
                padding="max_length", 
                max_length=256, 
                add_special_tokens=True,
                return_token_type_ids=self.return_token_type_ids,
            )
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
            )
            self.valid_data = self.valid_data.map(self.tokenize_data, batched=True)
            self.valid_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
            )
            print(self.train_data[0])

    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=4,
            pin_memory=True, 
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_data, 
            batch_size=self.batch_size, 
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )



if __name__ == "__main__":
    data_model = DataModule(version=1)
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
    print(data_model.tokenizer.decode(data_model.train_data[0]["input_ids"]))
