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
