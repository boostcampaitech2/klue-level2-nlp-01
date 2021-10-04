import torch
import os
import copy
from importlib import import_module
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.validation import compute_metrics_f1
from src.hyper_parameters import hp_space_sigopt
from src.data_load import *
from src.train import load_tokenizer_and_model, label_to_num

# 클로저 쓰자
def hp_space_sigopt(_):
    return [
        {
            "bounds": {"min": 5e-6, "max": 1e-4},
            "name": "learning_rate",
            "type": "double",
            "transformamtion": "log",
        },
        {"bounds": {"min": 1, "max": 6}, "name": "num_train_epochs", "type": "int"},
        {"bounds": {"min": 1, "max": 50}, "name": "seed", "type": "int"},
        {
            "categorical_values": ["4", "8", "16", "32", "64"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
        {"bounds": {"min": 100, "max": 1000}, "name": "warmup_steps", "type": "int"},
        {
            "bounds": {"min": 0.001, "max": 0.1},
            "name": "weight_decay",
            "type": "double",
        },
    ]


# 하이퍼 파라미터 튜닝용 설정
def set_hp_training_args(output_dir, train_args_cfg):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=train_args_cfg.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=train_args_cfg.eval_steps,  # evaluation step.
        disable_tqdm=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )


# 하이퍼 파라미터용 훈련
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
