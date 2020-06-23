import torch
import torchvision
from vision import get_labels, load_model, transform_image, predict
import torchvision.models as models
import subprocess
from PIL import Image

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def get_vision_models():
    # __dict__ This is the dictionary containing the module’s symbol table.
    model_names = sorted(name for name in models.__dict__ if
                         name.islower() and not name.startswith("__")  # and "inception" in name
                         and callable(models.__dict__[name]))
    return model_names


def makelogFile(name):
    subprocess.run(['sh', 'nvidiasmi.sh', name, '&'])


def stopNvidiaSmi():
    subprocess.run(['pkill', 'nvidia-smi'])


def main():
    # modelList = ['alexnet', 'densenet121','densenet161','densenet169','densenet201','googlenet',
    # 'inception_v3','mobilenet_v2','resnet101','shufflenet_v2_x2_0','mnasnet1_0','squeezenet1_0',
    # 'squeezenet1_1','wide_resnet50_2','wide_resnet101_2','vgg11','vgg11_bn','vgg13','vgg13_bn',
    # 'vgg16',
    # 'vgg19','vgg19_bn']
    model_list = get_vision_models()
    for model in model_list:
        modelrun(model)


def modelrun(model):
    vision_model = load_model(model)
    print("---- {} loaded -----".format(model))
    labels = get_labels()
    img = "../data/butterfly.jpg"
    makelogFile(model)
    for i in range(10000):
        image = Image.open(img)
        # image.show()
        image_tensor = transform_image(image)
        label = predict(vision_model, image_tensor, labels)
        #print("My guess is {} \n\n".format(label))
    stopNvidiaSmi()


if __name__ == "__main__":
    main()
