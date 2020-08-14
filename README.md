# Training and inference on computer vision models

<https://pytorch.org/docs/master/torchvision/models.html#classification>

`pip install -r requirements.txt`

## Get FLOPs

`python models.py`

Get FLOPs/FPOs using <https://github.com/Lyken17/pytorch-OpCounter>

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

## Get data from training

Download folder of images to train on  
`curl https://download.pytorch.org/tutorial/hymenoptera_data.zip --output /media/data/hymenoptera_data.zip`

Train models  
`cd Training`  
`python finetune_multiple.py`
