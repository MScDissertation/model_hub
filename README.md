# Environmental impact of computer vision

[![forthebadge](https://forthebadge.com/images/badges/powered-by-electricity.svg)](https://forthebadge.com)

# Analysis of data

## Notebooks

1. Training and FLOP analysis - get the data for model accuracy, power draw from power monitor readings and FLOPs. Linear regression is performed to get the relation between FLOPs and energy use.  
   Code for training in https://github.com/MScDisseration/model_hub/tree/master/Training  
   Code for FLOPs in https://github.com/MScDisseration/model_hub/blob/master/models.py

1. Vision models inference analysis - similar to previous notebook. Data is collected for power draw during inference (10000 inference run for each model).  
   Code for running inference in https://github.com/MScDisseration/model_hub/tree/master/Inference

## Training and inference on computer vision models

`pip install -r requirements.txt`

<https://pytorch.org/docs/master/torchvision/models.html#classification>

## Get Floating Point Operations, FLOPs

`python models.py`

Using <https://github.com/Lyken17/pytorch-OpCounter>

## Get data from training

Download folder of images to train on  
`curl https://download.pytorch.org/tutorial/hymenoptera_data.zip --output /media/data/hymenoptera_data.zip`

Train models  
`cd Training`  
`python finetune_multiple.py`

## Get power data for Inference

`cd Inference`  
`python multipleRuns.py`

## Extras

### Run inference on one model

`cd Inference`  
`python vision.py --path imagepath --model modelname`

E.g. `python vision.py --path "../data/butterfly.jpg" --model "alexnet"`

<!-- #### run tests

python -m unittest test_vision.py
python test_vision.py -->
