# Explainable Audio Representation Learning

## Abstract
Audio classification of sounds “in the wild”, i.e., in auditory environments in which
they would typically occur, remains a challenging yet relevant task. In this thesis, we
propose a dual-stream CNN architecture followed by a Label Embeddings Projection
(LEP) for audio classification. With these components, our network is able to approximate audio data while also harnessing semantic information from textual class label
embeddings. The contributions of this thesis are twofold: First, we improve upon the
state of the art in audio classification presented in Kazakos et al. (2021) with our
addition of the Label Embeddings Projection. Second, to introduce explainability
for our model, we also propose a gradient-based method for reconstructing the audio
that the network finds to be most salient.


## Preparation

* Requirements:
  * [PyTorch](https://pytorch.org) 1.7.1
  * [librosa](https://librosa.org): `conda install -c conda-forge librosa`
  * [h5py](https://www.h5py.org): `conda install h5py`
  * [wandb](https://wandb.ai/site): `pip install wandb`
  * [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
  * simplejson: `pip install simplejson`
  * psutil: `pip install psutil`
  * tensorboard: `pip install tensorboard` 
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/auditory-slow-fast/slowfast:$PYTHONPATH
```
* VGG-Sound:
  1. Download the audio. For instructions see [here](https://github.com/hche11/VGGSound)
  2. Download `train.pkl` ([link](https://www.dropbox.com/s/j60wkrcfdkfbvp9/train.pkl?dl=0)) and `test.pkl` ([link](https://www.dropbox.com/s/57rxp8wlgcqjbnd/test.pkl?dl=0)). I converted the original `train.csv` and `test.csv` (found [here](https://github.com/hche11/VGGSound/tree/master/data)) to pickle files with column names for easier use

## Training/validation on VGG-Sound
To train the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations 
```

To validate the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```
## Citation and special thanks

A base framework for the dual-stream CNN portion of the model architecture, as well as some of the documentation above, was created by:

Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen, **Slow-Fast Auditory Streams for Audio Recognition**, *ICASSP*, 2021

[Project's webpage](https://ekazakos.github.io/auditoryslowfast/)

[arXiv paper](https://arxiv.org/abs/2103.03516)

## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).

