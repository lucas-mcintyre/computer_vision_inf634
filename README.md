# Cleaning the Streets of Montevideo

## Understanding and "cleaning" street scenes in Montevideo, Uruguay

This repository contains the code and data for the project "Cleaning the Streets of Montevideo" by Thomas Wimmer and
Lucas McIntyre.
The project was done as part of the course "Computer Vision" at the École Polytechnique in the winter term 2022.

### Report and Slides

The report for the project is included in the repository [here](INF634-WIMMER-MCINTYRE.pdf).
The slides for the presentation are included in the repository [here](INF634-WIMMER-MCINTYRE-presentation.pdf).

### Demo Video

As the demo video is supposed to only be between 10 seconds and 1 minute long, we already ran some of the cells in the
notebook and showed the outputs of the cells. The video demonstrates the different tasks we can do with the pretrained
model; from evaluation of the detection results over the Grad-CAM visualizations to the actual cleaning of the streets
using image inpainting.

The demovideo can be found [here](INF634-WIMMER-MCINTYRE-demovideo.mp4).

### Setup

We recommend using Google Colab for running the notebooks provided in this repository.
In any case, we also provide a `environment.yml` file for setting up a conda environment with all required dependencies.

### Data

We recommend uploading the data to your Google Drive and mounting it in the notebooks.
The data can be downloaded from [here](https://www.kaggle.com/rodrigolaguna/clean-dirty-containers-in-montevideo).
We also provide a shared data folder under the
following [link](https://drive.google.com/drive/folders/1cI05rApTmmtGiYPJ4dbITSK2E9TdRagk?usp=sharing).

### Pretrained models

We provide some pretrained models under
this [link](https://drive.google.com/drive/folders/1-zKErO0z9cac12g59f69BF2FgeFhisDq?usp=sharing):
You can load them using the `load_model_from_saved` function in the `models.py` module.
The title of the model indicates important information needed to load the models, i.e.:

`[model_name]-[f1_score]-[fc1_neurons]-[fc2_neurons]-[...].pth`

### Notebooks

The following notebooks are provided in this repository:

- `Computer_Vision_INF634.ipynb`: This notebook contains code that can be used to train models, evaluate their
  performance, run the Grad-CAM on them and to digitally "clean" the images using image inpainting. The .py files in the
  `src` folder need to be uploaded to the colab environment for this notebook to work.
- `CycleGAN_Training.ipynb`: This notebook contains code for training the CycleGAN model (which unfortunately did not
  work very well).

### Code

The code for the project is provided in the `src` folder.
The following files are provided:

- `models.py`: This file contains the code for loading and creating models, training and saving them.
- `evaluation.py`: This file contains the code for evaluating the performance of the models (quantitatively and
  qualitatively).
- `data.py`: This file contains the code for loading the data and creating the data loaders.
- `interpretation.py`: This file contains the code for running Grad-CAM on the models.

(The code for street scene "cleaning" using image inpainting is given in the `Computer_Vision_INF634.ipynb` notebook.)