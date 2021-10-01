import pickle

import torch
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="klue/bert-base", batch_size=128, version=1):
        super().__init__()

        typed_entity_marker = ['[S:PER]', '[/S:PER]', '[S:ORG]', '[/S:ORG]',
                               '[O:PER]', '[/O:PER]', '[O:ORG]', '[/O:ORG]',
                               '[O:DAT]', '[/O:DAT]', '[O:LOC]', '[/O:LOC]',
                               '[O:POH]', '[/O:POH]', '[O:NOH]', '[/O:NOH]',]
        
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": typed_entity_marker})
        # typed-entity-marker를 위한 special token 추가
        # model 에 embedding layer 바꿔주기

        # if version == 1:
        #     self.preprocess = self.data_preprocess1
        #     # tokenize_data 함수에서 문장을 pair로 넣는 것과 하나로 넣는 것 차이 보정 필요.
        # else:
        #     self.preprocess = self.data_preprocess2

    def prepare_data(self):
        klue_re = load_dataset('klue', 're')
        self.train_data = klue_re['train']
        self.valid_data = klue_re['validation']
        
        with open('/opt/ml/klue/dict_klue_label_mapper.pkl', 'rb') as f:
            self.label_mapper = pickle.load(f)
        
        self.train_data = self.train_data.map(self.data_preprocess2, num_proc=4, remove_columns=["subject_entity", "object_entity"])
        self.valid_data = self.valid_data.map(self.data_preprocess2, num_proc=4, remove_columns=["subject_entity", "object_entity"])

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
        return self.tokenizer(
            # example["entity_span"],  # 이부분 전처리 방식에 따라 자동 보정 필요
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
                type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
            )
            self.valid_data = self.valid_data.map(self.tokenize_data, batched=True)
            self.valid_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
            )

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
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
    print(data_model.tokenizer.decode(data_model.train_data[0]["input_ids"]))
