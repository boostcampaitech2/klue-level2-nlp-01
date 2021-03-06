from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.utils.data import DataLoader
from src.data_load import *
import pandas as pd
import torch
import torch.nn.functional as F

from importlib import import_module
import pickle as pickle
import numpy as np
import os
from tqdm import tqdm


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def num_to_label(label, cfg):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    path = os.path.join(cfg.dir_path.code, "dict_num_to_label.pkl")
    origin_label = []
    with open(path, "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer, cfg):
    """
    test dataset을 불러온 후,
    전처리 후,
    tokenizing 합니다.
    """
    # 데이터셋 로드
    test_dataset = load_data(dataset_dir)

    # 전처리
    preprocess_name, is_two_sentence = cfg.preprocess.list[cfg.preprocess.pick]
    preprocess_func = getattr(import_module("src.pre_process"), preprocess_name)

    test_dataset, _, _ = preprocess_func(test_dataset)
    test_label = list(map(int, test_dataset["label"].values))

    # tokenizing dataset
    token_func = (
        "tokenized_dataset_with_division" if is_two_sentence else "tokenized_dataset"
    )
    tokenizer_module = getattr(import_module("src.tokenizing"), token_func)
    tokenized_test = tokenizer_module(test_dataset, tokenizer)

    return test_dataset["id"], tokenized_test, test_label


def inference_main(cfg):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    tokenizer_path = os.path.join(cfg.dir_path.base, cfg.name, cfg.dir_path.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ## load my model
    model_dir = os.path.join(
        cfg.dir_path.base, cfg.name, cfg.dir_path.model
    )  # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = cfg.dir_path.test_data_path
    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer, cfg
    )
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(
        model, Re_test_dataset, device
    )  # model에서 class 추론
    pred_answer = num_to_label(pred_answer, cfg)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    if not os.path.exists(
        os.path.join(cfg.dir_path.base, cfg.name, cfg.dir_path.submission)
    ):
        os.mkdir(os.path.join(cfg.dir_path.base, cfg.name, cfg.dir_path.submission))
    output.to_csv(
        os.path.join(
            cfg.dir_path.base, cfg.name, cfg.dir_path.submission, "submission.csv"
        ),
        index=False,
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("\n\n---- Finish! ----")
