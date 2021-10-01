import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers.utils.dummy_pt_objects import AutoModelForNextSentencePrediction


class Bertbase(pl.LightningModule):
    def __init__(self, model_name='klue/bert-base', lr=5e-5):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=30)
        self.num_classes = 30

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["label"]
        )

        preds = torch.argmax(outputs.logits, 1)
        self.log("train_loss", outputs.loss, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        val_acc = accuracy_score(preds.detach().cpu(), batch["label"].detach().cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", outputs.loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        return {"labels": labels, "logits": outputs.logits, "loss": outputs.loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)