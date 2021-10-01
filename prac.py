#%%
""" 코드 연습용 파일 """

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

tokenized_data = tokenizer(train_data['entity_span'], train_data['sentence'], return_tensors='pt',
                                padding=True, truncation=True, max_length=256, add_special_tokens=True)


#%%
from transformers import BertForPreTraining, BertModel


ids = tokenized_data["input_ids"][:2]
at_mask = tokenized_data["attention_mask"][:2]
t_ids = tokenized_data["token_type_ids"][:2]

outputs = sequence_model(input_ids=ids, attention_mask=at_mask, token_type_ids=t_ids)

print(outputs)


#%%


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



# %%
from transformers import BertTokenizerFast, RobertaModel, RobertaForSequenceClassification, RobertaConfig, AutoModel

model_name = "klue/roberta-base"

config = RobertaConfig.from_pretrained(model_name)

roberta = AutoModel.from_pretrained(model_name)
roberta_for_sequence = RobertaForSequenceClassification.from_pretrained(model_name)
roberta_large = AutoModel.from_pretrained("klue/roberta-large")

roberta_tokenizer = BertTokenizerFast.from_pretrained(model_name)
roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)

#%%
import pickle
bert = AutoModelForSequenceClassification.from_pretrained('klue/bert-large', num_labels=2)
type_embeddings = bert.bert.embeddings.token_type_embeddings.weight

with open("./bert_type_embeddings.pkl", "wb") as f:
    pickle.dump(type_embeddings, f)

# %%

from transformers import ElectraModel, ElectraTokenizer, AutoModelForSequenceClassification, AutoTokenizer

electra = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
electra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


# %%
print(roberta_tokenizer.tokenize("아버지가방에들어가신다."))
print(roberta_tokenizer.tokenize("이순신은조선중기의무신이다."))
print(electra_tokenizer.tokenize("아버지가방에들어가신다."))
print(electra_tokenizer.tokenize("이순신은조선중기의무신이다."))
#%%

# inputs = roberta_tokenizer("아버지가방에들어가신다.", "이순신은조선중기의무신이다람쥐는삐약삐약병아리는음메.", padding=True, max_length=256, return_tensors="pt")
# roberta(**inputs)

# %%
param_groups = {"no_decay": [], "decay": []}
for name, param in bert.named_parameters():
    if "bias" in name or "LayerNorm.weight" in name:
        param_groups["no_decay"].append(param)
    else:
        param_groups["decay"].append(param)

import torch

optimizer = torch.optim.Adam([{"params":param_groups["decay"], "weight_decay": 0.01}, {"params":param_groups["no_decay"], "weight_decay": 0.0}], lr=0.001)
