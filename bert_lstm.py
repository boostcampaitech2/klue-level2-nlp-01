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

# 출력 부분에 LSTM 적용한 버트 모델 정의
class BertLSTM(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.config =  AutoConfig.from_pretrained(MODEL_NAME)
        self.config.num_labels = 30
        self.num_labels = 30
        
        Model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=self.config)
        
        self.Bert = Model.bert        
        self.lstm = nn.LSTM(input_size=768,
                    hidden_size=768,
                    num_layers=1,
                    bidirectional=False,
                    batch_first=True)
        self.dropout = Model.dropout
        self.classifier = Model.classifier

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.zeros((1, 5, 768)).to(device)  # (num_layers * num_dirs, B, d_h)
        self.c_0 = torch.zeros((1, 5, 768)).to(device)  # (num_layers * num_dirs, B, d_h)
        
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
        
        lstm_outputs, _ = self.lstm(outputs[0], (self.h_0, self.c_0))
        # hidden_outputs = outputs[0]
        # pooled_output = outputs[1]
        
        # lstm_outputs[:,-1,:] : context vector
        pooled_output = self.dropout(lstm_outputs[:,-1,:])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        
        outputs = (loss, logits)
        return outputs

def trainLSTM():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BertLSTM(MODEL_NAME)
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results/lstm_results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=1500,                 # model saving step.
        num_train_epochs=20,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=5,  # batch size per device during training
        per_device_eval_batch_size=5,   # batch size for evaluation
        warmup_steps=1500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 1500,            # evaluation step.
        load_best_model_at_end = True,
        fp16=True
    )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model/lstm')

def main():
    showUtilization()
    trainLSTM()

if __name__ == '__main__':
    main()

