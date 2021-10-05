import pandas as pd

from src.EDA_AEDA import AEDA_generator


def preprocess_decorator(func):
    """
    데코레이션으로 사용할때는 Arguments를 (dataset, set_AEDA=False, AEDA_cfg=None) 로 받게 하세요.
    리턴값은 sentences, labels, add_tokens, special_tokens 로 통일 해주세요.
    최종적으로 전처리된dataset, add_tokens, add_special_tokens의 인자를 리턴합니다.
    """

    def inner_logics(dataset, set_AEDA=False, AEDA_cfg=None):
        ids = []
        sentences = []
        labels = []
        add_tokens = set()
        special_tokens = set()
        for sub, obj, sen, label, id in zip(
            dataset["subject_entity"],
            dataset["object_entity"],
            dataset["sentence"],
            dataset["label"],
            dataset["id"],
        ):
            sens, a_t, s_t = func(sub, obj, sen)
            sentences.append(sens)
            labels.append(label)
            ids.append(id)
            add_tokens.update(a_t)
            special_tokens.update(s_t)
        if set_AEDA:
            aeda_sentences = AEDA_generator(sentences, AEDA_cfg)
            repetition = AEDA_cfg.generator.repetition
            aeda_labels = [value for value in labels for _ in range(repetition)]
            add_ids = [value for value in ids for _ in range(repetition)]
            sentences.extend(aeda_sentences)
            labels.extend(aeda_labels)
            ids.extend(add_ids)
        out_dataset = pd.DataFrame(
            {
                "id": ids,
                "sentence": sentences,
                "label": labels,
            }
        )
        out_dataset = out_dataset.drop_duplicates(
            subset=["sentence", "label"],
            keep="first",
        )
        return (
            out_dataset,
            list(add_tokens),
            {"additional_special_tokens": list(special_tokens)},
        )

    return inner_logics


@preprocess_decorator
def dataset_preprocess_base(sub, obj, sen):
    """
    sentence: 이순신@PER[SEP]장군@TITLE[SEP]@@@@@이순신 장군은 조선 제일의 무신이다.
    """
    add_tokens, special_tokens = [], []
    pre_sen = (
        sub["word"]
        + "@"
        + sub["type"]
        + "[SEP]"
        + obj["word"]
        + "@"
        + obj["type"]
        + "@@@@@"
    )
    add_tokens.extend([sub["type"], obj["type"]])
    sentence = pre_sen + sen
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def dataset_preprocess_base_for_AEDA(sub, obj, sen):
    """
    sentence: 이순신@PER & 장군@TITLE @@@@@ 이순신 장군은 조선 제일의 무신이다.
    @@@@@ 를 쓸 경우 토크나이저시 해당부분이 SEP으로 교체됩니다.
    AEDA 시 좋은성능을 보일거라 예상합니다.
    """
    add_tokens, special_tokens = [], []
    pre_sen = (
        sub["word"]
        + "@"
        + sub["type"]
        + " & "
        + obj["word"]
        + "@"
        + obj["type"]
        + "@@@@@"
    )
    add_tokens.extend([sub["type"], obj["type"]])
    sentence = pre_sen + sen
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def dataset_preprocess_with_special_tokens(sub, obj, sen):
    """
    [CLS] [S:PER] 이순신 [/S] 장군은 조선 제일의  [O:TITLE] 무신 [/O] 이다.[SEP]
    """
    add_tokens, special_tokens = [], []
    sub_start_token, sub_end_token = f'[S:{sub["type"]}]', "[/S]"
    obj_start_token, obj_end_token = f'[O:{obj["type"]}]', "[/O]"
    special_tokens.update([sub_start_token, obj_start_token])
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
    sentence = temp
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def dataset_with_full_special_tokens(sub, obj, sen):
    """
    스페셜 토큰을 추가합니다. (꽤 많음..)
    [CLS] [SUB:PER] 이순신 [/SUB:PER] 장군은 조선 제일의  [OBJ:TITLE] 무신 [/OBJ:TITLE] 이다.[SEP]
    """
    add_tokens, special_tokens = [], []
    sub_start_token, sub_end_token = f'[S:{sub["type"]}]', "[/S]"
    obj_start_token, obj_end_token = f'[O:{obj["type"]}]', "[/O]"

    sub_start_token, sub_end_token = f'[SUB:{sub["type"]}]', f'[/SUB:{sub["type"]}]'
    obj_start_token, obj_end_token = f'[OBJ:{obj["type"]}]', f'[/OBJ:{obj["type"]}]'
    special_tokens.extend(
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
    sentence = temp
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def dataset_with_least_special_tokens(sub, obj, sen):
    """
    tokenizer에 따라 sentence를 tokenizing 합니다.
    스페셜 토큰을 추가합니다. ([SUB],[/SUB],[OBJ],[/OBJ] 네 가지)
    [CLS] [SUB] PER @ 이순신 [/SUB] 장군은 조선 제일의 [OBJ] TITLE @ 무신 [/OBJ] 이다.[SEP]
    """
    add_tokens, special_tokens = [], []
    sub_start_token, sub_end_token = "[SUB]", "[/SUB]"
    obj_start_token, obj_end_token = "[OBJ]", "[/OBJ]"

    add_index = [
        (sub["start_idx"], sub_start_token + f" {sub['type']} @ "),
        (sub["end_idx"] + 1, sub_end_token),
        (obj["start_idx"], obj_start_token + f" {obj['type']} @ "),
        (obj["end_idx"] + 1, obj_end_token),
    ]
    add_index.sort(key=lambda x: x[0], reverse=True)
    special_tokens.extend([sub["type"], obj["type"]])

    temp = sen
    for token in add_index:
        temp = temp[: token[0]] + f" {token[1]} " + temp[token[0] :]
    sentence = temp
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def data_preprocess2(sub, obj, sen):
    """typed-entity-marker"""
    add_tokens, special_tokens = [], []
    tokens = [
        (sub["start_idx"], f"[S:{sub['type']}]"),
        (sub["end_idx"] + 1, f"[/S:{sub['type']}]"),
        (obj["start_idx"], f"[O:{obj['type']}]"),
        (obj["end_idx"] + 1, f"[/O:{obj['type']}]"),
    ]
    special_tokens.extend(
        [
            f"[S:{sub['type']}]",
            f"[/S:{sub['type']}]",
            f"[O:{obj['type']}]",
            f"[/O:{obj['type']}]",
        ]
    )
    tokens.sort(key=lambda x: x[0], reverse=True)
    for start_index, token in tokens:
        sen = "".join([sen[:start_index], token, sen[start_index:]])
    sentence = sen
    return sentence, add_tokens, special_tokens


@preprocess_decorator
def data_preprocess_Cap_Token(sub, obj, sen):
    """
    [CLS] [SUB:PER] 이순신 [/SUB:PER] 장군은 조선 제일의 [OBJ:TITLE] 무신 [/OBJ:TITLE] 이다.[SEP]
    스페셜 토큰은 다음과 같이 생성됩니다.
    ["SUB", "OBJ", "PER", "TITLE" ...]
    """
    add_tokens, special_tokens = [], ["SUB", "OBJ", sub["type"], obj["type"]]
    ###### TODO Your Logic ######
    sub_start_token, sub_end_token = f"[SUB:{sub['type']}]", f"[/SUB:{sub['type']}]"
    obj_start_token, obj_end_token = f"[OBJ:{obj['type']}]", f"[/OBJ:{obj['type']}]"
    add_index = [
        (sub["start_idx"], sub_start_token + f" {sub['type']} @ "),
        (sub["end_idx"] + 1, sub_end_token),
        (obj["start_idx"], obj_start_token + f" {obj['type']} @ "),
        (obj["end_idx"] + 1, obj_end_token),
    ]
    add_index.sort(key=lambda x: x[0], reverse=True)
    temp = sen
    for token in add_index:
        temp = temp[: token[0]] + f" {token[1]} " + temp[token[0] :]
    sentence = temp

    return sentence, add_tokens, special_tokens


@preprocess_decorator
def data_preprocess_template(sub, obj, sen):
    """
    데이터 전처리 탬플릿입니다. 복붙해서 사용하세요
    만들 데이터 전처리 함수에 대한 설명을 입력해주세요.
    sub, obj는 하나의 객체입니다. 데코레이션 함수에 의해 알아서 for구문 돌립니다.
    """
    add_tokens, special_tokens = [], []
    ###### TODO Your Logic ######

    #############################
    sentence = ""
    return sentence, add_tokens, special_tokens
