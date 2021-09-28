#%%
import pickle

import torch
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification

model_name = 'klue/bert-base'

klue_re = load_dataset('klue', 're')
train_data = klue_re['train']
valid_data = klue_re['validation']

with open('/opt/ml/klue/dict_label_to_num.pkl', 'rb') as f:
    l2n = pickle.load(f)

with open('/opt/ml/klue/dict_num_to_label.pkl', 'rb') as f:
    n2l = pickle.load(f)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model =  AutoModelForPreTraining.from_pretrained(model_name)
sequence_model = AutoModelForSequenceClassification.from_pretrained(model_name)


def extract_entity_word(example):
    return {"entity_span": example["subject_entity"]["word"] + " [SEP] " + example["object_entity"]["word"]}
    

train_data = train_data.map(extract_entity_word, num_proc=4, remove_columns=["subject_entity", "object_entity"])

tokenized_data = tokenizer(train_data['entity_span'], train_data['sentence'], batched=True, return_tensors='pt',
                                padding=True, truncation=True, max_length=256, add_special_tokens=True)

train_label = 


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


#%%
import pickle

import torch
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification

model_name = 'klue/bert-base'

klue_re = load_dataset('klue', 're')
train_data = klue_re['train']
valid_data = klue_re['validation']

with open('/opt/ml/klue/dict_label_to_num.pkl', 'rb') as f:
    l2n = pickle.load(f)

with open('/opt/ml/klue/dict_num_to_label.pkl', 'rb') as f:
    n2l = pickle.load(f)