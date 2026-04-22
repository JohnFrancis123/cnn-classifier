# CNN Classifier

A PyTorch-based image classification project that trains a compact convolutional neural network on CIFAR-10 and serves predictions through a FastAPI API.

This repository includes:

- a reproducible training pipeline
- a small CNN architecture for 32x32 RGB images
- a FastAPI inference service with file upload support
- a Typer CLI for training and serving
- tests and lint configuration for local development

## Overview

The model is trained on CIFAR-10, which contains 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The training pipeline automatically downloads CIFAR-10, applies augmentation to the training split, creates a deterministic 80/20 train-validation split, and saves the best checkpoint based on validation accuracy.

## Model

The classifier is a compact CNN with:

- 3 convolution blocks
- batch normalization and ReLU activations
- max pooling in the first two blocks
- adaptive average pooling before classification
- a final linear layer that outputs 10 logits

This keeps the project simple enough for learning and portfolio use while still covering a realistic end-to-end ML workflow.

## Project Structure

```text
cnn-classifier/
├── configs/
│   └── default.yaml
├── checkpoints/
│   └── best.pt
├── data/
├── src/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   └── train.py
└── tests/
```

## Requirements

- Python 3.10+
- pip

Core dependencies are defined in `pyproject.toml`, including PyTorch, TorchVision, FastAPI, Typer, Pillow, and Uvicorn.

## Installation

Clone the repo and install it in editable mode.

```bash
git clone https://github.com/your-username/cnn-classifier.git
cd cnn-classifier
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e .[dev]
```

## Configuration

Default training settings live in `configs/default.yaml`:

```yaml
dataset: cifar10
data_dir: "./data"
batch_size: 64
epochs: 80
lr: 0.001
seed: 42
checkpoint_dir: "./checkpoints"
num_workers: 2
```

You can change these values directly or point the CLI at another YAML file.

## Training

Run training with the default config:

```bash
python -m src.cli train
```

Run training with a custom config file:

```bash
python -m src.cli train --config configs/default.yaml
```

What happens during training:

- CIFAR-10 is downloaded automatically if it is not already present
- the training set uses random crop and horizontal flip augmentation
- the validation and test sets use deterministic preprocessing
- the best checkpoint is saved to `checkpoints/best.pt`

## Inference API

Start the API server with the default checkpoint:

```bash
python -m src.cli serve
```

Or specify a checkpoint, host, and port:

```bash
python -m src.cli serve --checkpoint checkpoints/best.pt --host 127.0.0.1 --port 8000
```

Once running, FastAPI docs are available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

### Prediction Endpoint

`POST /predict`

Accepts one uploaded image file and returns the predicted class with a confidence score.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-F "file=@path/to/image.jpg"
```

Example response:

```json
{
	"prediction": "frog",
	"confidence": 0.9834
}
```

Possible error responses:

- `503 Model not loaded` if the server starts without a valid checkpoint
- `400 Image required` if the uploaded file is not an image
- `400` with an error message if image decoding or inference fails

## Development

Run linting:

```bash
python -m ruff check src tests
```

Run tests:

```bash
python -m pytest tests -vv --tb=short
```

## Why This Repo Exists

This project is a good template for a small but complete ML application because it combines:

- model definition
- data loading and preprocessing
- training and checkpointing
- an inference API
- automated tests

It works well as a portfolio project, a starter repo for image classification experiments, or a clean reference for structuring a small PyTorch service.
