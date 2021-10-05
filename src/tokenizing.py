def tokenized_dataset(dataset, tokenizer):
    """
    기본 토크나이저 생성입니다.
    """
    return tokenizer(
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )


def tokenized_dataset_with_division(dataset, tokenizer):
    """
    2 문장을 나누어 토크나이저에 입력합니다.
    문장 사이에는 @@@@@가 들어가도록 전처리 해야합니다.
    """
    concat_entity = []
    sentences = []
    for sen in dataset["sentence"]:
        f_sen, l_sen = sen.split("@@@@@")
        concat_entity.append(f_sen)
        sentences.append(l_sen)
    return tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
