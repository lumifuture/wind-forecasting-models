# wind-forecasting-models

**Shared PyTorch model definitions** for the wind forecasting system.  Used across both training and inference services in a polyrepo architecture.

## What's in this repo?

This package contains reusable, production-ready implementations of key deep learning architectures:

-  `LSTMModel` — vanilla LSTM for time series forecasting
-  `CNNLSTMModel` — 1D CNN for feature extraction + LSTM
-  `TransformerModel` — attention-based forecasting model
-  `model_registry.py` — dynamic model resolution by type string

## Directory Structure

```plaintext

wind-forecasting-models/
├── wind_models/
│   ├── lstm.py                # LSTMModel class
│   ├── cnn_lstm.py            # CNN + LSTM hybrid
│   ├── transformer.py         # Transformer for sequence modeling
│   └── model_registry.py      # get_model_class("lstm") -> LSTMModel
├── tests
│   └── test_lstm.py           # Sanity check for LSTM model
├── setup.py                   # Python package definition
├── requirements.txt
└── README.md
````

## Quickstart

### 1. Install

Clone this repo locally:

```bash
git clone https://github.com/lumifuture/wind-forecasting-models
cd wind-forecasting-models
pip install -e .
````

### 2. Use in Code

```python
from wind_models.lstm import LSTMModel

model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
```

Or use dynamically:

```python
from wind_models.model_registry import get_model_class

ModelClass = get_model_class("lstm")
model = ModelClass(input_size=1, hidden_size=64, num_layers=2)
```

## Testing

```bash
pytest tests/
```

Make sure you install `pytest` first:

```bash
pip install pytest
```

## Supported Model Types

| Model Type    | Import Path                  | Description                      |
| ------------- | ---------------------------- | -------------------------------- |
| `lstm`        | `wind_models.lstm.LSTMModel` | Vanilla LSTM                     |
| `cnn_lstm`    | `wind_models.cnn_lstm.*`     | 1D CNN + LSTM hybrid             |
| `transformer` | `wind_models.transformer.*`  | Self-attention-based forecasting |

Use `model_registry.get_model_class(model_type)` for safe resolution.

## Used By

| Repository                   | Purpose                   |
| ---------------------------- | ------------------------- |
| `wind-forecasting-training`  | Model training and saving |
| `wind-forecasting-inference` | Model loading and serving |

## Dev Tips

* All models inherit from `torch.nn.Module`
* To load trained models with metadata, ensure the model’s `__init__()` accepts parameters saved in `metadata.json` (e.g., `input_size`, `hidden_size`)
* This repo should **not contain training logic** — that lives in the training service

## License

[Apache License 2.0](https://github.com/lumifuture/wind-forecasting-models/blob/main/LICENSE)

