# pytorch models

https://pytorch.org/docs/master/torchvision/models.html#classification

Get FLOPs/FPOs using

https://github.com/Swall0w/torchstat

https://github.com/Lyken17/pytorch-OpCounter

pip install -r requirements.txt

### Get FPO

python models.py

### Run inference on one model

cd Inference
python vision.py --path imagepath --model modelname

E.g. python vision.py --path "../data/butterfly.jpg" --model "alexnet"

## Get power data

python multipleRuns.py

#### run tests

python -m unittest test_vision.py
python test_vision.py
