import torch
import os
import copy
import optuna
from importlib import import_module
from shutil import copyfile
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.validation import compute_metrics, compute_metrics_f1
from src.data_load import *
from src.util import label_to_num


def optuna_hp_func_with_cfg(hp_cfg):
    def hp_space_optuna(trial: optuna.trial.Trial):
        return {
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", hp_cfg.learning_rate.min, hp_cfg.learning_rate.max
            ),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs",
                hp_cfg.num_train_epochs.min,
                hp_cfg.num_train_epochs.max,
            ),
            "seed": trial.suggest_int("seed", hp_cfg.seed.min, hp_cfg.seed.max),
            "warmup_steps": trial.suggest_int(
                "warmup_steps", hp_cfg.warmup_step.min, hp_cfg.warmup_step.max
            ),
            "weight_decay": trial.suggest_uniform(
                "weight_decay", hp_cfg.weight_decay.min, hp_cfg.weight_decay.max
            ),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps",
                hp_cfg.gradient_accumulation_steps.min,
                hp_cfg.gradient_accumulation_steps.max,
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", hp_cfg.per_device_train_batch_size
            ),
        }

    return hp_space_optuna


# 하이퍼 파라미터 튜닝용 설정
def set_hp_training_args(output_dir, train_args_cfg, name):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=train_args_cfg.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=train_args_cfg.eval_steps,  # evaluation step.
        fp16=train_args_cfg.fp16,
        disable_tqdm=True,
        per_device_train_batch_size=train_args_cfg.batch_size,  # batch size per device during training
        per_device_eval_batch_size=train_args_cfg.batch_size,  # batch size for evaluation
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=name,
    )


# 하이퍼 파라미터용 훈련
def hyper_parameter_train(cfg):
    train_name = cfg.name
    print(f"\n\n *** {train_name} HyperParameter START !! ***\n\n")

    # 토크나이저와 모델 불러오기
    MODEL_INDEX = cfg.model.pick
    print(f"\n\n Target model: {cfg.model.model_list[MODEL_INDEX]}\n\n")
    tokenizer, model = load_tokenizer_and_model(MODEL_INDEX, cfg.model)

    # 데이터셋 불러오기
    train_dataset = train_data_with_addition(
        cfg.dir_path.train_data_path, cfg.hyperparameters.dataset.train.add_data
    )
    if cfg.dataset.dev_data:
        valid_dataset = train_data_with_addition(
            cfg.dir_path.dev_data_path, cfg.hyperparameters.dataset.valid.add_data
        )
    else:
        train_dataset, valid_dataset = get_stratified_K_fold(
            train_dataset, cfg.hyperparameters.dataset.valid
        )

    # 데이터 전처리

    print(f"\n START PreProcess \n")
    preprocess_name, is_two_sentence = cfg.preprocess.list[cfg.preprocess.pick]
    preprocess_func = getattr(import_module("src.pre_process"), preprocess_name)

    train_dataset, add_tokens, special_tokens = preprocess_func(
        train_dataset, cfg.EDA_AEDA.AEDA.set, cfg.EDA_AEDA.AEDA
    )
    train_dataset = train_dataset.sample(frac=1)  # 훈련데이터 뒤섞기

    valid_dataset, _, _ = preprocess_func(valid_dataset)

    train_label = label_to_num(train_dataset["label"].values, cfg)
    valid_label = label_to_num(valid_dataset["label"].values, cfg)

    print(f"\n\n train_data: {len(train_dataset)}, dev_data: {len(valid_dataset)} \n\n")

    # 데이터셋 토크나이징
    print(f"\n\n tokenizing START \n\n")

    tokenizer.add_tokens(add_tokens)
    tokenizer.add_special_tokens(special_tokens)

    token_func = (
        "tokenized_dataset_with_division" if is_two_sentence else "tokenized_dataset"
    )
    tokenizer_module = getattr(import_module("src.tokenizing"), token_func)

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

    def custom_model_init():
        return getattr(
            import_module("src.custom_models"),
            cfg.model.custom_model_args.list[cfg.model.custom_model_args.pick],
        )(cfg.model.model_list[MODEL_INDEX])

    output_dir = os.path.join(cfg.dir_path.base, train_name, "hp_optimize")

    # Training 설정
    training_args = set_hp_training_args(output_dir, cfg.train_args, cfg.name)
    trainer = Trainer(
        model_init=model_init,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        tokenizer=tokenizer,
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # hp train model
    trainer.hyperparameter_search(
        direction="maximize",
        backend=cfg.hyperparameters.backend,
        compute_objective=compute_metrics_f1,
        hp_space=optuna_hp_func_with_cfg(cfg.hyperparameters.cfg),
        n_trials=cfg.hyperparameters.n_trials,
    )
