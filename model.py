import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from utils import compute_metrics, draw_confusion_matrix
from data import DataModule

class ModelModule(pl.LightningModule):
    def __init__(self, model_name='klue/bert-base', lr=5e-5, dirpath=None, version=2):
        super().__init__()

        self.num_classes = 30
        self.lr = lr
        self.dirpath = dirpath

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.best_f1_score = 1

        if version == 1 :
            # baseline 방식
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=30)
        else :
            # typed-entity-marker 방식
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=30)
            self.model.resize_token_embeddings(self.model.config.vocab_size + len(DataModule.typed_entity_marker))


    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.model(
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
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["label"]
        )

        preds = torch.argmax(outputs.logits, 1)
        self.log("eval_loss", outputs.loss, prog_bar=True)
        
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        
        preds = torch.argmax(logits, 1)
        probs = torch.softmax(logits, dim=-1)

        # auprc , classification report , confusion matrix 는 numpy 입력을 받음.
        labels = labels.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        f1_score, auprc, acc = compute_metrics(preds, probs, labels)

        self.log("eval_micro f1 score", f1_score, prog_bar=True)
        self.log("eval_auprc", auprc, prog_bar=True)
        self.log("eval_acc", acc, prog_bar=True)

        # checkpoint directory에 best model의 confusion matrix와 classification report 저장
        if f1_score >= self.best_f1_score :
            self.best_f1_score = f1_score
            print(f"save best result ... best f1 score: {self.best_f1_score}")
            classification_result = classification_report(labels, preds, zero_division=1)
            draw_confusion_matrix(labels, preds, self.dirpath, num_classes=30)
            with open(f"{self.dirpath}/classification_result_of_best_model.txt", "w") as f:
                f.write(classification_result)

    def configure_optimizers(self):
        param_groups = {"no_decay": [], "decay": []}
        for name, param in self.named_parameters():
            if "bias" in name or "LayerNorm.weight" in name:
                param_groups["no_decay"].append(param)
            else:
                param_groups["decay"].append(param)
        
        optimizer = AdamW(
            [{"params":param_groups["decay"], "weight_decay": 0.01}, 
             {"params":param_groups["no_decay"], "weight_decay": 0.0}], lr=self.lr # 5e-5
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=50, 
            num_training_steps=2520, 
            num_cycles=3
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(set_to_none=True)
