#%%
import pickle
import torch
from operator import itemgetter

from datasets import load_dataset


with open("/opt/ml/pstage/dict_label_to_num.pkl", "rb") as f:
    l2n = pickle.load(f)


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


def load_data(dataset_dir, version=2):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    dataset = load_dataset('csv', data_files=[dataset_dir])['train']
    
    if version == 1 :
        func = data_preprocess1
    elif version == 2 :
        func = data_preprocess2
    else :
        raise ValueError("Version 1=baseline approach , 2=typed-entity-marker approach")
    
    dataset = dataset.map(func, num_proc=4, remove_columns=["id", "subject_entity", "object_entity", "source"])
    return dataset


def data_preprocess1(example):
    """ baseline approach (entity span) """
    example["label"] = l2n[example["label"]]
    example["subject_entity"] = eval(example["subject_entity"])
    example["object_entity"] = eval(example["object_entity"])
    return {"entity_span": example["subject_entity"]["word"] + "[SEP]" + example["object_entity"]["word"]}


def data_preprocess2(example):
    """ typed-entity-marker """
    sentence = example["sentence"]
    example["label"] = l2n[example["label"]]
    example["subject_entity"] = eval(example["subject_entity"])
    example["object_entity"] = eval(example["object_entity"])
    sub_w, sub_s, sub_e, sub_t = example["subject_entity"].values()
    ob_w, ob_s, ob_e, ob_t = example["object_entity"].values()
    tokens = [
      (sub_s, f'[S:{sub_t}]'),
      (sub_e+1, f'[/S:{sub_t}]'),
      (ob_s, f'[O:{ob_t}]'),
      (ob_e+1, f'[/O:{ob_t}]'),
    ]
    tokens.sort(key=lambda x: x[0], reverse=True)
    for start_index, token in tokens:
      sentence = ''.join([sentence[:start_index], token, sentence[start_index:]])
    example["sentence"] = sentence
    return example


def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        ) 
    return tokenized_sentences
