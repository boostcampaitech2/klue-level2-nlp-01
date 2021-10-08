import os
import pickle


def label_to_num(label, cfg):
    path = os.path.join(cfg.dir_path.code, "dict_label_to_num.pkl")
    num_label = []
    with open(path, "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label
