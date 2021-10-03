def tokenized_dataset(dataset, tokenizer):
    """
    tokenizer에 따라 sentence를 tokenizing 합니다.
    [CLS]이순신@PER[SEP]장군@TITLE[SEP]이순신 장군은 조선 제일의 무신이다.[SEP]
    """
    concat_entity = []
    type_set = set()  # ORG, DAT 등 등록안된 토큰 추가
    for sub, obj in zip(dataset["subject_entity"], dataset["object_entity"]):
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
    special_token_set = set(["[/S]", "[/O]"])

    for sub, obj, sen in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        sub_start_token, sub_end_token = f'[S:{sub["type"]}]', "[/S]"
        obj_start_token, obj_end_token = f'[O:{obj["type"]}]', "[/O]"
        special_token_set.update([sub_start_token, obj_start_token])
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
    [CLS][SUB] PER@ 이순신 [/SUB] 장군은 조선 제일의 [OBJ] TITLE@ 무신 [/OBJ] 이다.[SEP]
    """
    concat_entity = []
    addtional_token_set = set()
    sub_start_token, sub_end_token = "[SUB]", "[/SUB]"
    obj_start_token, obj_end_token = "[OBJ]", "[/OBJ]"

    for sub, obj, sen in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
    ):
        add_index = [
            (sub["start_idx"], sub_start_token + f" {sub['type']} @ "),
            (sub["end_idx"] + 1, sub_end_token),
            (obj["start_idx"], obj_start_token + f" {obj['type']} @ "),
            (obj["end_idx"] + 1, obj_end_token),
        ]
        add_index.sort(key=lambda x: x[0], reverse=True)
        addtional_token_set.update([sub["type"], obj["type"]])

        temp = sen
        for token in add_index:
            temp = temp[0 : token[0]] + f" {token[1]} " + temp[token[0] :]
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
