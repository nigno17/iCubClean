# iCubClean

This repository contains the CNN used to predict the cleaning trajectories in the paper:

_N.Cauli, P.Vicente, J.Kim, B.Damas, A.Bernardino, F.Cavallo, and J.Santos-Victor, “Autonomous table-cleaning from kinesthetic demonstrations using Deep Learning,” in Joint IEEE International Conference on Development and Learning (ICDL) and Epigenetic Robotics (EpiRob), IEEE, 2018._

The network is able to predict, from images, mean vectors and covariance matrices of a mixture of gaussians that represent 2D cleaning trajectories. The Gaussian Mixture Regression (GMR) algorithm is used to extract the estimation of the cleaning trajectories. To have more information about the architecture please refer to the [ICDL-EpiRob 2018 paper](http://vislab.isr.ist.utl.pt/wp-content/uploads/2018/07/ncauli_icdl2018.pdf). If you use any of this code, please cite this publication.

## Dependencies

In order to run the code you need to install few dependencies:

* Python 2.7
* [Pytorch 0.4.0](https://pytorch.org/) (recommended the CUDA version)
* [Visdom](https://github.com/facebookresearch/visdom)
* [opencv](https://opencv.org/)

## Usage

First clone the repository:

```bash
git clone https://github.com/nigno17/iCubClean.git
```

Second download and unzip the repository inside the folder datasetICDL-JINT (if you want more information about the dataset please go to the [link](http://vislab.isr.ist.utl.pt/datasets/#clean2) and refer to  ___Robot Table Cleaning v2___):

```bash
cd iCubClean
mkdir datasetICDL-JINT
cd datasetICDL-JINT
wget http://soma.isr.ist.utl.pt/vislab_data/cleaning_tasks/datasetICDL-JINT.zip
unzip datasetICDL-JINT.zip
rm datasetICDL-JINT.zip
```
Run the training:

```bash
cd <local_path_to_the_repository>
python CleaningNetwork.py
```

## Visdom

If you want to visualize the loss training and validation error through the epochs and an example of a bach of images used for training run a visdom server:

```bash
python -m visdom.server
```
Then open in a browser the following link: http://localhost:8097
