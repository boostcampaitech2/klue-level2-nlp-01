import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


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


def load_tokenizer_and_model(model_index, model_cfg):
    MODEL_NAME = model_cfg.model_list[model_index]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = model_cfg.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    return tokenizer, model


def preprocessing_dataset(dataset):
    """
    처음 불러온 csv 파일을 속성을 포함한 형태의 DataFrame으로 변경 시켜줍니다.
    subject_entity와 object_entity를 Dict 형태로 불러옵니다.
    """
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


def additional_data(train_data_path):
    """
    entity의 속성을 이용하여 추가데이터를 제작합니다.
    """
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
    add_data = load_data(train_data_path)
    add_data = add_data.rename(columns=config["change_entity"])
    # 추가 데이터를 만들 수 있는 라벨만 남긴다
    add_data = add_data[add_data.label.isin(config["remain_label_list"])]
    # 속성 정렬을 해준다 (정렬을 안할경우 obj와 sub의 순서가 바뀌어 보기 불편함)
    add_data = add_data[config["cols"]]
    # 서로 반대되는 뜻을 가진 라벨을 바꿔준다.
    add_data = add_data.replace({"label": config["change_values"]})
    return add_data


def train_data_with_addition(train_data_path, is_add_data):
    """
    additional 데이터를 추가한 훈련데이터를 불러옵니다.
    """
    if not is_add_data:
        return load_data(train_data_path)
    added_data = load_data(train_data_path).append(additional_data(train_data_path))
    added_data["subject_entity_string"] = added_data["subject_entity"].astype(str)
    added_data["object_entity_string"] = added_data["object_entity"].astype(str)
    added_data = added_data.drop_duplicates(
        subset=["subject_entity_string", "object_entity_string", "sentence"],
        keep="first",
    )  # 중복데이터 제거
    return added_data


def get_stratified_K_fold(dataset: pd.DataFrame, dataset_cfg):
    """
    stratified_K_fold를 구현하였습니다.
    n_splits은 몇개로 나눌것인지를 의미합니다.
    num은 n_splits으로 나눴을때 몇번째 데이터를 가져올지를 의미합니다.
    shuffle은 랜덤하게 나눕니다. 이때는 num을 사용하지않고 무조건 1번째것을 가져옵니다.
    """
    assert dataset_cfg.n_splits > dataset_cfg.num, "num은 n_splits보다 작아야 합니다."
    skf = StratifiedKFold(n_splits=dataset_cfg.n_splits, shuffle=dataset_cfg.shuffle)
    k_fold_data = list(skf.split(dataset, dataset["label"]))
    if dataset_cfg.shuffle or dataset_cfg.num == 0:
        train_index, valid_index = k_fold_data[0][0], k_fold_data[0][1]
    else:
        train_index, valid_index = (
            k_fold_data[dataset_cfg.num][0],
            k_fold_data[dataset_cfg.num][1],
        )
    return dataset.iloc[train_index], dataset.iloc[valid_index]


if __name__ == "__main__":
    print(additional_data())
