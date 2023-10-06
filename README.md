# image_enhancement
This is just for fun now and it is for image enhancing.

# Image Enhancement with Deep Learning

This project aims to enhance images using deep learning techniques. It includes a Python implementation of a convolutional neural network (CNN) for image enhancement. The codebase provides functionalities for training, testing, and evaluating the model on image datasets. The project also supports multiple loss functions for customization.

## Project Structure

- `main.py`: Main script for training and testing the image enhancement model.
- `model.py`: Defines the architecture of the image enhancement generator model.
- `loss_functions.py`: Custom loss functions for PSNR, SSIM, LNL1, and a combined loss.
- `data_loader.py`: Custom dataset class and data loaders for loading and augmenting data.
- `train.py`: Training loop for the deep learning model.
- `test.py`: Testing and evaluation of the trained model.
- `sample_plt.py`: Functions for displaying random sample results.
- `preprocess.py`: Script for preprocessing image data before training.
- `train.py`: Script for training the deep learning model.
- `test.py`: Script for testing and evaluating the trained model.
- `gif_enhancer.py`: Script for enhancing GIF files using the trained model.
- `image_enhancer.py`: Script for enhancing single images using the trained model.

## Script Details

### `main.py`

- Purpose: Entry point for training and testing the image enhancement model.
- Functionality:
  - Handles command-line arguments for training and testing modes.
  - Initializes the model, data loaders, and other necessary components.
  - Executes training or testing based on user inputs.

### `model.py`

- Purpose: Defines the architecture of the image enhancement generator model.
- Functionality:
  - Implements the generator model using PyTorch.
  - Provides options for both complex and simple generator architectures.
  - Defines forward pass logic for enhancing input images.

### `loss_functions.py`

- Purpose: Contains custom loss functions for training the image enhancement model.
- Functionality:
  - Defines various loss functions, including PSNR, SSIM, LNL1, and a combined loss.
  - Calculates loss values for model training and evaluation.

### `data_loader.py`

- Purpose: Manages data loading and augmentation for the training and testing datasets.
- Functionality:
  - Defines a custom dataset class for reading and preprocessing image data.
  - Implements data augmentation techniques for training data.
  - Provides data loaders for efficiently loading batches of data.

### `train.py`

- Purpose: Implements the training loop for the deep learning model.
- Functionality:
  - Loads the training dataset using data loaders from `data_loader.py`.
  - Optimizes the model using a specified loss function and optimizer.
  - Logs training progress and saves checkpoints during training.

### `test.py`

- Purpose: Performs testing and evaluation of the trained deep learning model.
- Functionality:
  - Loads the test dataset using data loaders from `data_loader.py`.
  - Computes evaluation metrics such as PSNR, SSIM, VIF, and LNL1.
  - Saves enhanced images and evaluation results.

### `sample_plt.py`

- Purpose: Contains functions for displaying random sample results.
- Functionality:
  - Randomly selects and visualizes input and enhanced images.
  - Provides a visual assessment of the model's performance.

### `preprocess.py`

- Purpose: Preprocesses image data before training the deep learning model.
- Functionality:
  - Resizes images to specified target sizes or fractions.
  - Utilizes multiprocessing for parallel processing of images.

### `gif_enhancer.py`

- Purpose: Enhances GIF files using the trained model.
- Functionality:
  - Reads input GIF frames and applies enhancement iteratively.
  - Saves enhanced GIFs as output.

### `image_enhancer.py`

- Purpose: Enhances single images using the trained model.
- Functionality:
  - Reads input images and applies enhancement iteratively.
  - Saves enhanced images as output.

## Getting Started

1. **Prerequisites**: Python 3.x and PyTorch. Install dependencies using `pip`.

2. **Dataset Preparation**: Organize image data as shown in the directory structure.

3. **Training**: Run `train.py`. Adjust hyperparameters in the script.

4. **Testing**: Test using the best checkpoint saved during training with `test.py`.

5. **GIF Enhancement**: Enhance GIF files using `gif_enhancer.py`.

6. **Image Enhancement**: Enhance single images using `image_enhancer.py`.

7. **Visualization**: Visualize random test samples using `sample_plt.py`.

## Model Architecture

Two generator models for image enhancement:
- `Generator`: Complex model with encoder and decoder components.
- `SimpleGenerator`: Simpler model with fewer layers.

## Loss Functions

Various loss functions for training:
- `PSNRLoss`: Peak Signal-to-Noise Ratio loss.
- `TangentSSIMLoss`: Structural Similarity Index (SSIM) loss.
- `LNL1Loss`: Logarithmic Normalized L1 loss.
- `PSNR_SSIM_LNL1Loss`: Combined loss using PSNR, SSIM, and LNL1.

## Results

Enhanced images and evaluation metrics in the `results/` directory.

## Acknowledgments

- Custom loss functions based on `torchmetrics` library.
- Dataset sourced from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from ETH Zurich.
- Dataset structure follows DIV2K dataset format.

Feel free to customize and extend this project for your image enhancement tasks.
