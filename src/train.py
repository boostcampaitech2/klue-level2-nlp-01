import pickle as pickle
import torch
import os
import copy

from importlib import import_module
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.validation import compute_metrics, compute_metrics_f1
from src.hyper_parameters import hp_space_sigopt
from src.data_load import *


def label_to_num(label, cfg):
    path = os.path.join(cfg.dir_path.code, "dict_label_to_num.pkl")
    num_label = []
    with open(path, "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def load_tokenizer_and_model(model_index, model_cfg):
    MODEL_NAME = model_cfg.model_list[model_index]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = model_cfg.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    return tokenizer, model


def set_hp_training_args(output_dir, train_args_cfg):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=train_args_cfg.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=train_args_cfg.eval_steps,  # evaluation step.
        disable_tqdm=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )


def set_training_args(output_dir, log_dir, train_args_cfg):
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    return TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=train_args_cfg.save_total_limit,  # number of total save model.
        save_steps=train_args_cfg.save_steps,  # model saving step.
        num_train_epochs=train_args_cfg.num_train_epochs,  # total number of training epochs
        learning_rate=train_args_cfg.learning_rate,  # learning_rate
        per_device_train_batch_size=train_args_cfg.batch_size,  # batch size per device during training
        per_device_eval_batch_size=train_args_cfg.batch_size,  # batch size for evaluation
        warmup_steps=train_args_cfg.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=train_args_cfg.weight_decay,  # strength of weight decay
        logging_dir=log_dir,  # directory for storing logs
        logging_steps=train_args_cfg.logging_steps,  # log saving step.
        evaluation_strategy=train_args_cfg.evaluation_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=train_args_cfg.eval_steps,  # evaluation step.
        load_best_model_at_end=True,
        report_to="wandb",
    )


def hyper_parameter_train(cfg):
    train_name = cfg.name

    # load model and tokenizer
    MODEL_INDEX = cfg.model.pick
    tokenizer, model = load_tokenizer_and_model(MODEL_INDEX, cfg.model)

    # load dataset
    dataset = train_data_with_addition(cfg.dir_path.train_data_path, cfg.dataset)
    temp_dataset, train_dataset = get_stratified_K_fold(
        dataset, cfg.hyperparameters.dataset.train
    )
    valid_cfg = cfg.hyperparameters.dataset.valid
    valid_cfg.n_splits = int(
        valid_cfg.n_splits
        * (cfg.hyperparameters.dataset.train.n_splits - 1)
        / cfg.hyperparameters.dataset.train.n_splits
    )
    _, valid_dataset = get_stratified_K_fold(temp_dataset, valid_cfg)
    train_label = label_to_num(train_dataset["label"].values, cfg)
    valid_label = label_to_num(valid_dataset["label"].values, cfg)

    # tokenizing dataset
    tokenizer_module = getattr(
        import_module("src.tokenizing"), cfg.tokenizer.list[cfg.tokenizer.pick]
    )
    tokenized_train = tokenizer_module(train_dataset, tokenizer)
    tokenized_valid = tokenizer_module(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # setting model hyperparameter
    MODEL_NAME = cfg.model.model_list[MODEL_INDEX]
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = cfg.model.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))

    def model_init():
        return copy.deepcopy(model)

    output_dir = os.path.join(cfg.dir_path.base, train_name, "hp_optimize")

    # Training 설정
    training_args = set_hp_training_args(output_dir, cfg.train_args)
    trainer = Trainer(
        model_init=model_init,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics_f1,  # define metrics function
    )

    # hp train model
    trainer.hyperparameter_search(
        direction="maximize",
        backend=cfg.hyperparameters.backend,
        hp_space=hp_space_sigopt,
        n_trials=cfg.hyperparameters.n_trials,
    )


def model_train(cfg):
    train_name = cfg.name

    # load model and tokenizer
    MODEL_INDEX = cfg.model.pick
    tokenizer, model = load_tokenizer_and_model(MODEL_INDEX, cfg.model)

    # load dataset
    dataset = train_data_with_addition(cfg.dir_path.train_data_path, cfg.dataset)
    train_dataset, valid_dataset = get_stratified_K_fold(dataset, cfg.dataset)
    train_label = label_to_num(train_dataset["label"].values, cfg)
    valid_label = label_to_num(valid_dataset["label"].values, cfg)

    # tokenizing dataset
    tokenizer_module = getattr(
        import_module("src.tokenizing"), cfg.tokenizer.list[cfg.tokenizer.pick]
    )
    tokenized_train = tokenizer_module(train_dataset, tokenizer)
    tokenized_valid = tokenizer_module(valid_dataset, tokenizer)

    tokenizer.save_pretrained(
        os.path.join(cfg.dir_path.base, train_name, cfg.dir_path.tokenizer)
    )

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # setting model hyperparameter
    MODEL_NAME = cfg.model.model_list[MODEL_INDEX]
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = cfg.model.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.parameters  # 이건 도대체 무슨역할일까?
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
    model.save_pretrained(
        os.path.join(cfg.dir_path.base, train_name, cfg.dir_path.model)
    )


def train(cfg):
    if cfg.hyperparameters.set:
        hyper_parameter_train(cfg)
    else:
        model_train(cfg)
