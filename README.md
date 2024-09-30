# Vehicle Damage Detection

This project implements a deep learning model for detecting and classifying vehicle damage from images. It uses a Faster R-CNN model with a ResNet50 backbone for object detection, trained on a custom dataset of vehicle damage images.

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)
8. [Contributing](#contributing)
9. [License](#license)

## Requirements

This project uses Poetry for dependency management. The main dependencies are:

- Python 3.8+
- PyTorch 2.4.1+
- torchvision 0.19.1+
- matplotlib 3.4.3+
- tqdm 4.62.3+
- Pillow 8.3.2+
- pycocotools 2.0.2+
- torchmetrics 1.4.2+

For a complete list of dependencies, see the `pyproject.toml` file.

## Setup

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install the project dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Set up your environment variables by creating a `.env` file in the project root directory. Use the provided `.env.example` as a template.

6. To start training run the `training_pipeline.py` file and to evaluate the model run `eval.py`

## Project Structure

```
vehicle-damage-detection/
│
├── vehicledamagedetection/
│   ├── __init__.py
│   ├── dataset_class.py
│   ├── utils.py
│   ├── train.py
│   └── eval.py
├── tests/
├── .env
├── pyproject.toml
├── README.md
└── runs/
    └── vehicle_damage_detection/
```

## Usage

To train the model:

```bash
poetry run python vehicledamagedetection/train.py
```

To evaluate the model:

```bash
poetry run python vehicledamagedetection/eval.py
```

## Training

The training script (`train.py`) does the following:

1. Loads the dataset using the custom `VehicleDamageDataset` class.
2. Initializes the Faster R-CNN model with a ResNet50 backbone.
3. Trains the model for a specified number of epochs. (30 used)
4. Saves the best model based on validation accuracy.
5. Logs training progress and metrics using TensorBoard.

You can modify training parameters in the `.env` file.

## Evaluation

The evaluation script (`eval.py`) performs the following:

1. Loads the trained model.
2. Runs the model on the test dataset.
3. Computes evaluation metrics such as mAP (mean Average Precision).

## Visualization

To visualize training progress and results:

1. Start TensorBoard:
   ```bash
   poetry run tensorboard --logdir=runs --port=6006
   ```
2. Open a web browser and go to `http://localhost:6006`.

If port 6006 is already in use, you can specify a different port:
```bash
poetry run tensorboard --logdir=runs --port=6007
```

## Training Curve

![alt text](/training_loss.png)

## Example predictions

![alt text](/inference_1.png)
![alt text](/inference_2.png)
