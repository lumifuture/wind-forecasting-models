from wind_models.lstm import LSTMModel
from wind_models.cnn_lstm import CNNLSTMModel
from wind_models.transformer import TransformerModel


def get_model_class(model_type: str):
    model_map = {
        "lstm": LSTMModel,
        "cnn_lstm": CNNLSTMModel,
        "transformer": TransformerModel,
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_map[model_type]
