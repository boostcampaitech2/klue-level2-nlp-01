import pickle

import torch
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils.dummy_pt_objects import AutoModelForPreTraining

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="klue/bert-base", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        klue_re = load_dataset('klue', 're')
        self.train_data = klue_re['train']
        self.valid_data = klue_re['validation']
        
        with open('/opt/ml/klue/dict_label_to_num.pkl', 'rb') as f:
            self.l2n = pickle.load(f)
        
        with open('/opt/ml/klue/dict_num_to_label.pkl', 'rb') as f:
            self.n2l = pickle.load(f)

        self.train_data.map(self.data_preprocess, num_proc=4, remove_columns=["subject_entity", "object_entity"])
        self.valid_data.map(self.data_preprocess, num_proc=4, remove_columns=["subject_entity", "object_entity"])

    def data_preprocess(self, example): 
        return {"entity_span": example["subject_entity"]["word"] + "[SEP]" + example["object_entity"]["word"]}
        

    def tokenize_data(self, example):
        return self.tokenizer(
            example["entity_span"], 
            example["sentence"], 
            truncation=True, 
            padding="max_length", 
            max_length=256, 
            add_special_tokens=True
        )
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )
            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )