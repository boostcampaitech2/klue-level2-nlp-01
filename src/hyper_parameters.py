from ray import tune


def ray_hp_space():
    return {
        "learning_rate": tune.loguniform(5e-6, 5e-4),
        "num_train_epochs": tune.choice(range(1, 6)),
        "seed": tune.choice(range(1, 42)),
    }


def apply_hyper_parameters(trainer):
    trainer.hyperparameter_search(
        direction="maximize",  # NOTE: or direction="minimize"
        hp_space=ray_hp_space,  # NOTE: if you wanna use optuna, change it to optuna_hp_space
        backend="ray",  # NOTE: if you wanna use optuna, remove this argument
    )
