import pickle as pickle
import os
import pandas as pd
import torch
from basecode.load_data import RE_Dataset, load_data as ld_base

TRAIN_DATA_PATH = "/opt/ml/dataset/train/train.csv"
TEST_DATA_PATH = "/opt/ml/dataset/test/test_data.csv"

def load_train_data():
  """ pandas 형식의 훈련 데이터 """
  return ld_base(TRAIN_DATA_PATH)

def load_test_data():
  """ pandas 형식의 테스트 데이터 """
  return ld_base(TEST_DATA_PATH)

def additional_data():
  config = {
    "change_entity": {"subject_entity": "object_entity", "object_entity": "subject_entity"},
    "remain_label_list": ['no_relation', 'org:members', 'org:alternate_names',
                  'per:children', 'per:alternate_names', 'per:other_family', 
                  'per:colleagues', 'per:siblings', 'per:spouse',
                  'org:member_of', 'per:parents'],
    "change_values": { "org:member_of": "org:members", 
                      "org:members": "org:member_of", 
                      "per:parents": "per:children", 
                      "per:children": "per:parents" },
    "cols": ['id', 'sentence', 'subject_entity', 'object_entity', 'label']
  }

  add_data = load_train_data().rename(columns=config['change_entity']) # 훈련 데이터를 불러오고 subject_entity와 object_entity만 바꾼다.
  add_data = add_data[add_data.label.isin(config['remain_label_list'])] # 추가 데이터를 만들 수 있는 라벨만 남긴다
  add_data = add_data[config["cols"]] # 속성 정렬을 해준다 (정렬을 안할경우 obj와 sub의 순서가 바뀌어 보기 불편함)
  add_data = add_data.replace({ "label": config['change_values'] }) # 서로 반대되는 뜻을 가진 라벨을 바꿔준다.
  return add_data 

def train_data_with_addition():
  return load_train_data().append(additional_data())

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
      )
  return tokenized_sentences

if __name__ == "__main__":
  print(len(train_data_with_addition()))
