from transformers import AutoTokenizer
from src.pre_process import dataset_with_full_special_tokens
from src.data_load import *

path = "/opt/ml/dataset/train/dev.csv"
dataset = load_data(path)


class cfg:
    add_data = True


add_dataset = train_data_with_addition(path, cfg())
len(dataset), len(add_dataset)

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


class ge_cfg:
    p = 0.9
    repetition = 1


class ad_cfg:
    morpheme_analyzer = "Okt"
    tqdm = True
    punc_ratio = 0.3
    punctuations = [".", ",", "!", "?", ";", ":"]
    generator = ge_cfg()


a = ["아버지가 방에 들어가신다.", "어머니가 방에서 나오신다"]


to_data = dataset_with_full_special_tokens(dataset, True, ad_cfg)
print(to_data)
