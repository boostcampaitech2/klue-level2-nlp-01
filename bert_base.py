import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Bertbase(pl.LightningModule):
    def __init__(self, model_name='klue/bert-base', lr=5e-5):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 30)
        self.num_classes = 30

    def forward(self, input_ids, attention_mask):

        pass

    def training_step(self, batch, batch_idx):

        pass

    def validation_step(self, batch, batch_idx):

        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])