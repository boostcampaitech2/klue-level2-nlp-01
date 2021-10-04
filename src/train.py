import pickle as pickle
import torch
import os
from importlib import import_module
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.validation import compute_metrics
from src.data_load import *
from src.hyper_parameters import hyper_parameter_train
from src.util import label_to_num

# Train용 설정
def set_training_args(output_dir, log_dir, train_args_cfg):
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    return TrainingArguments(
        output_dir=output_dir,  # output directory
        logging_dir=log_dir,  # directory for storing logs
        save_total_limit=train_args_cfg.save_total_limit,  # number of total save model.
        save_steps=train_args_cfg.save_steps,  # model saving step.
        num_train_epochs=train_args_cfg.num_train_epochs,  # total number of training epochs
        learning_rate=train_args_cfg.learning_rate,  # learning_rate
        per_device_train_batch_size=train_args_cfg.batch_size,  # batch size per device during training
        per_device_eval_batch_size=train_args_cfg.batch_size,  # batch size for evaluation
        warmup_steps=train_args_cfg.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=train_args_cfg.weight_decay,  # strength of weight decay
        logging_steps=train_args_cfg.logging_steps,  # log saving step.
        evaluation_strategy=train_args_cfg.evaluation_strategy,  # evaluation strategy to adopt during training
        fp16=train_args_cfg.fp16,
        dataloader_pin_memory=train_args_cfg.dataloader_pin_memory,
        gradient_accumulation_steps=train_args_cfg.gradient_accumulation_steps,
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=train_args_cfg.eval_steps,  # evaluation step.
        load_best_model_at_end=True,
        disable_tqdm=train_args_cfg.disable_tqdm,
        # wandb 저장
        report_to="wandb",
    )


def model_train(cfg):
    train_name = cfg.name
    print(f"\n\n *** {train_name} START !! ***\n\n")
    # 토크나이저와 모델 불러오기
    MODEL_INDEX = cfg.model.pick
    print(f"\n\n Target model: {cfg.model.model_list[MODEL_INDEX]}\n\n")
    tokenizer, model = load_tokenizer_and_model(MODEL_INDEX, cfg.model)

    # 데이터셋 불러오기
    train_dataset = train_data_with_addition(
        cfg.dir_path.train_data_path, cfg.dataset.add_data
    )
    if cfg.dataset.dev_data:
        valid_dataset = train_data_with_addition(
            cfg.dir_path.dev_data_path, cfg.dataset.add_dev_data
        )
    else:
        train_dataset, valid_dataset = get_stratified_K_fold(train_dataset, cfg.dataset)
    train_label = label_to_num(train_dataset["label"].values, cfg)
    valid_label = label_to_num(valid_dataset["label"].values, cfg)

    print(f"\n\n train_data: {len(train_dataset)}, dev_data: {len(valid_dataset)} \n\n")

    # 데이터셋 토크나이징
    print(f"\n\n tokenizer_module: {cfg.tokenizer.list[cfg.tokenizer.pick]}\n\n")
    tokenizer_module = getattr(
        import_module("src.tokenizing"), cfg.tokenizer.list[cfg.tokenizer.pick]
    )
    tokenized_train = tokenizer_module(train_dataset, tokenizer)
    tokenized_valid = tokenizer_module(valid_dataset, tokenizer)

    tokenizer.save_pretrained(
        os.path.join(cfg.dir_path.base, train_name, cfg.dir_path.tokenizer)
    )

    # 파이토치 데이터셋 제작
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # 모델 파라미터 설정
    if cfg.model.custom_model:
        model = getattr(
            import_module("src.custom_models"),
            cfg.model.custom_model_args.list[cfg.model.custom_model_args.pick],
        )
    else:
        MODEL_NAME = cfg.model.model_list[MODEL_INDEX]
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = cfg.model.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )
    model.resize_token_embeddings(len(tokenizer))
    # model.parameters  # 이건 도대체 무슨역할일까?
    model.to(device)

    # Training 설정
    output_dir = os.path.join(cfg.dir_path.base, train_name, cfg.train_args.output)
    log_dir = os.path.join(cfg.dir_path.base, train_name, cfg.train_args.log)
    training_args = set_training_args(output_dir, log_dir, cfg.train_args)
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()

    # 모델 저장
    model.save_pretrained(
        os.path.join(cfg.dir_path.base, train_name, cfg.dir_path.model)
    )


def train(cfg):
    if cfg.hyperparameters.set:
        # 하이퍼파라미터 트레이닝
        hyper_parameter_train(cfg)
    else:
        # 모델 트레이닝
        model_train(cfg)
