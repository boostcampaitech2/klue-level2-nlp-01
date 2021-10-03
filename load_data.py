import pickle as pickle
import os
import pandas as pd
from pandas.core.frame import DataFrame
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

TRAIN_DATA_PATH = "/opt/ml/dataset/train/train.csv"
TEST_DATA_PATH = "/opt/ml/dataset/test/test_data.csv"


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 속성을 포함한 형태의 형태의 DataFrame으로 변경 시켜줍니다."""
    import ast

    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = ast.literal_eval(i)
        j = ast.literal_eval(j)

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_train_data():
    """pandas 형식의 훈련 데이터"""
    return load_data(TRAIN_DATA_PATH)


def load_test_data():
    """pandas 형식의 테스트 데이터"""
    return load_data(TEST_DATA_PATH)


def additional_data():
    config = {
        "change_entity": {
            "subject_entity": "object_entity",
            "object_entity": "subject_entity",
        },
        "remain_label_list": [
            "no_relation",
            "org:members",
            "org:alternate_names",
            "per:children",
            "per:alternate_names",
            "per:other_family",
            "per:colleagues",
            "per:siblings",
            "per:spouse",
            "org:member_of",
            "per:parents",
            "org:top_members/employees",
        ],
        "change_values": {
            "org:member_of": "org:members",
            "org:members": "org:member_of",
            "per:parents": "per:children",
            "per:children": "per:parents",
            "org:top_members/employees": "per:employee_of",
        },
        "cols": ["id", "sentence", "subject_entity", "object_entity", "label"],
    }

    # 훈련 데이터를 불러오고 subject_entity와 object_entity만 바꾼다.
    add_data = load_train_data().rename(columns=config["change_entity"])
    # 추가 데이터를 만들 수 있는 라벨만 남긴다
    add_data = add_data[add_data.label.isin(config["remain_label_list"])]
    # 속성 정렬을 해준다 (정렬을 안할경우 obj와 sub의 순서가 바뀌어 보기 불편함)
    add_data = add_data[config["cols"]]
    # 서로 반대되는 뜻을 가진 라벨을 바꿔준다.
    add_data = add_data.replace({"label": config["change_values"]})
    return add_data


def train_data_with_addition():
    added_data = load_train_data().append(additional_data())
    added_data["subject_entity_string"] = added_data["subject_entity"].astype(str)
    added_data["object_entity_string"] = added_data["object_entity"].astype(str)
    added_data = added_data.drop_duplicates(
        subset=["subject_entity_string", "object_entity_string", "sentence"],
        keep="first",
    )  # 중복데이터 제거
    return added_data


def get_stratified_K_fold(dataset: pd.DataFrame, shuffle=False, n_splits=7, num=0):
    assert n_splits > num, "num은 n_splits보다 작아야 합니다."
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    k_fold_data = list(skf.split(dataset, dataset["label"]))
    if shuffle or num == 0:
        train_index, valid_index = k_fold_data[0][0], k_fold_data[0][1]
    else:
        train_index, valid_index = k_fold_data[num][0], k_fold_data[num][1]
    return dataset.iloc[train_index], dataset.iloc[valid_index]


def tokenized_dataset(dataset, tokenizer):
    """
    tokenizer에 따라 sentence를 tokenizing 합니다.
    [CLS]이순신@PER[SEP]장군@TITLE[SEP]이순신 장군은 조선 제일의 무신이다.
    """
    concat_entity = []
    type_set = set()  # ORG, DAT 등 등록안된 토큰 추가
    for sub, obj, sen in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        temp = ""
        temp = (
            sub["word"] + "@" + sub["type"] + "[SEP]" + obj["word"] + "@" + obj["type"]
        )
        type_set.update([sub["type"], obj["type"]])
        concat_entity.append(temp)
    tokenizer.add_tokens(list(type_set))
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def tokenized_dataset_with_special_tokens(dataset, tokenizer):
    """
    tokenizer에 따라 sentence를 tokenizing 합니다.
    스페셜 토큰을 추가합니다.
    [CLS][S:PER]이순신[/S] 장군은 조선 제일의 [O:TITLE]무신[/O]이다.[SEP]
    """
    concat_entity = []
    special_token_set = set()

    for sub, obj, sen in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        sub_start_token, sub_end_token = f'[S:{sub["type"]}]', "[/S]"
        obj_start_token, obj_end_token = f'[O:{obj["type"]}]', "[/O]"
        special_token_set.update(
            [sub_start_token, sub_end_token, obj_start_token, obj_end_token]
        )
        add_index = [
            (sub["start_idx"], sub_start_token),
            (sub["end_idx"] + 1, sub_end_token),
            (obj["start_idx"], obj_start_token),
            (obj["end_idx"] + 1, obj_end_token),
        ]
        add_index.sort(key=lambda x: x[0], reverse=True)
        temp = sen
        for token in add_index:
            temp = temp[0 : token[0]] + f" {token[1]} " + temp[token[0] :]
        concat_entity.append(temp)
    special_token = {"additional_special_tokens": list(special_token_set)}
    tokenizer.add_special_tokens(special_token)
    tokenized_sentences = tokenizer(
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def tokenized_dataset_with_least_special_tokens(dataset, tokenizer):
    """
    tokenizer에 따라 sentence를 tokenizing 합니다.
    스페셜 토큰을 추가합니다.
    [CLS][SUB] PER@이순신 [/SUB] 장군은 조선 제일의 [OBJ]TITLE@무신[/OBJ]이다.[SEP]
    """
    concat_entity = []
    addtional_token_set = set()
    sub_start_token, sub_end_token = "[SUB]", "[/SUB]"
    obj_start_token, obj_end_token = "[OBJ]", "[/OBJ]"

    for sub, obj, sen in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        add_index = [
            (sub["start_idx"], sub_start_token + f" {sub['type']}@"),
            (sub["end_idx"] + 1, sub_end_token + " "),
            (obj["start_idx"], obj_start_token + f" {obj['type']}@"),
            (obj["end_idx"] + 1, obj_end_token + " "),
        ]
        add_index.sort(key=lambda x: x[0], reverse=True)
        addtional_token_set.update([sub["type"], obj["type"]])

        temp = sen
        for token in add_index:
            temp = temp[0 : token[0]] + f" {token[1]}" + temp[token[0] :]
        concat_entity.append(temp)
    special_token = {
        "additional_special_tokens": ["[SUB]", "[/SUB]", "[OBJ]", "[/OBJ]"]
    }
    tokenizer.add_tokens(list(addtional_token_set))
    tokenizer.add_special_tokens(special_token)
    tokenized_sentences = tokenizer(
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


if __name__ == "__main__":
    from transformers import AutoTokenizer

    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train = tokenized_dataset_with_least_special_tokens(
        train_data_with_addition(), tokenizer
    )
    k = tokenizer.convert_ids_to_tokens(tokenized_train[0].ids)
    print(k)
    print(len(tokenizer))
