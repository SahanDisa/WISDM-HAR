def get_model_config(model_name):
    configs = {
        "MLP": {
            "layers": [128, 64, 32],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
        "CNN": {
            "filters": [32, 64, 128],
            "kernel_size": [3, 3],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
        "RNN": {
            "hidden_units": 64,
            "num_layers": 2,
            "activation": "tanh",
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
        },
        "LSTM": {
            "hidden_units": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
        "TCN": {
            "num_filters": 64,
            "kernel_size": 3,
            "num_layers": 4,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
    }
    return configs.get(model_name, f"Configuration for '{model_name}' not found.")