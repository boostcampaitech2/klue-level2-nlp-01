from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers
from torch.utils.data import TensorDataset
from load_data import *
from models import *
from utils import *

def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": TokenClassificationModel.typed_entity_marker})
    
    # load dataset
    train_dataset = load_data2("./train.csv", './dict_label_to_num.pkl' )
    dev_dataset = load_data2("./dev.csv",  './dict_label_to_num.pkl' )

    tokenized_train = tokenized_dataset2(train_dataset, tokenizer, 150, None)
    tokenized_dev = tokenized_dataset2(dev_dataset,tokenizer, 150, None)
    
    #train
    train_input_ids = torch.tensor([f.input_ids for f in tokenized_train], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in tokenized_train], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in tokenized_train], dtype=torch.long)
    train_e1_mask = torch.tensor([f.e1_mask for f in tokenized_train], dtype=torch.float) 
    train_e2_mask = torch.tensor([f.e2_mask for f in tokenized_train], dtype=torch.float) 
    train_label_ids = torch.tensor([f.label_id for f in tokenized_train], dtype=torch.long)
    RE_train_dataset = TensorDataset(train_input_ids, train_input_mask,train_segment_ids, train_label_ids, train_e1_mask, train_e2_mask)
    
    #valid
    val_input_ids = torch.tensor([f.input_ids for f in tokenized_dev], dtype=torch.long)
    val_input_mask = torch.tensor([f.input_mask for f in tokenized_dev], dtype=torch.long)
    val_segment_ids = torch.tensor([f.segment_ids for f in tokenized_dev], dtype=torch.long)
    val_e1_mask = torch.tensor([f.e1_mask for f in tokenized_dev], dtype=torch.long)
    val_e2_mask = torch.tensor([f.e2_mask for f in tokenized_dev], dtype=torch.long)
    val_label_ids = torch.tensor([f.label_id for f in tokenized_dev], dtype=torch.long)
    RE_dev_dataset = TensorDataset(val_input_ids, val_input_mask,val_segment_ids, val_label_ids, val_e1_mask, val_e2_mask)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    # model = SequenceClassificationModel(MODEL_NAME)
    model = TokenClassificationModel(MODEL_NAME)
    model.to(device)
    
    param_groups = {"no_decay": [], "decay": []}
    for name, param in model.named_parameters():
        if "bias" in name or "LayerNorm.weight" in name:
            param_groups["no_decay"].append(param)
        else:
            param_groups["decay"].append(param)

    optimizer = transformers.AdamW([{"params":param_groups["decay"], "weight_decay": 0.01}, {"params":param_groups["no_decay"], "weight_decay": 0.0}], lr=5e-5)
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=1890)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=1260, num_cycles=3)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=20,                 # model saving step.
        num_train_epochs=10,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        #fp16=True,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=4,
        warmup_steps=50,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=20,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 20,            # evaluation step.
        load_best_model_at_end = True
    )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        optimizers=(optimizer, scheduler),
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model')

def main(args):
    train(args)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--version", type=int, help="1: origin , 2: type-entity-marker", default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    main(args)
