import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from train import *
import warnings
warnings.filterwarnings('ignore')
from GPUtil import showUtilization
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool

class FeatureExtractionBert(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.config =  AutoConfig.from_pretrained(MODEL_NAME)
        self.Bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=self.config).bert
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.Bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

def FeatureExtractor(feature_extractor, dataloader):
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    feature_extractor.eval()
    outputs = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs.append(feature_extractor(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                ).pooler_output)
    return torch.cat(outputs).detach().cpu().numpy()

def PrepareFeatures():
    # 가장 점수 높았던 베이스라인 모델 불러오기
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    features = FeatureExtractionBert('./results/checkpoint-2500').to(device)

    # 특성 추출할 데이터 로드
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")
    dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    train_dataloader = DataLoader(RE_train_dataset, batch_size=16, shuffle=False)
    dev_dataloader = DataLoader(RE_dev_dataset, batch_size=16, shuffle=False)

    ml_train = FeatureExtractor(features, train_dataloader)
    ml_valid = FeatureExtractor(features, dev_dataloader)

    return ml_train, ml_valid, train_label, dev_label

def pca(ml_train, ml_valid):
    pca = PCA(n_components=0.95)
    ml_train_reduced = pca.fit_transform(ml_train)
    ml_valid_reduced = pca.transform(ml_valid)
    return ml_train_reduced, ml_valid_reduced

def train():
    ml_train, ml_valid, train_label, valid_label = PrepareFeatures()
    ml_train_reduced, ml_valid_reduced = pca(ml_train, ml_valid)

    train_data = Pool(data=ml_train_reduced, label=train_label)
    valid_data = Pool(data=ml_valid_reduced, label=valid_label)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cat_reduced = CatBoostClassifier(task_type="GPU", devices=device)
    model_cat_reduced.fit(train_data, eval_set=valid_data, use_best_model=True, early_stopping_rounds=100, verbose=100)

    y_pred_valid = model_cat.predict(ml_valid)
    print(f'Accuracy: {accuracy_score(valid_data, y_pred_valid)}, f1_score: {f1_score(valid_data, y_pred_valid, average="micro")}')

def main():
    train()

if __name__ == "__main__":
    main()

