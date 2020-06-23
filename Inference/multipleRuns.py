import torch
import torchvision
from vision import get_labels, load_model, transform_image, predict
import torchvision.models as models
import subprocess
from PIL import Image
import datetime
import os
import uuid

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
    subprocess.Popen(['sh', 'nvidiasmi.sh', name])  # make it non blocking


def stopNvidiaSmi():
    subprocess.run(['pkill', 'nvidia-smi'])


def prep_file(fileName):
    path = '../logs/' + fileName
    if os.path.exists(path):
        id = uuid.uuid4()
        renameFile = '../logs/' + fileName.split(".")[0] + str(id) + '.csv'
        os.rename(path, renameFile)
    with open(path, "w") as file1:
        row = "model" + "," + "start_time" + "," + "end_time" + "\n"
        file1.writelines(row)


def main():
    fileName = 'pm.csv'
    prep_file(fileName)
    pmfile = '../logs/' + fileName
    model_list = get_vision_models()
    for model in model_list:
        modelrun(model, pmfile)


def modelrun(model, pmfile):
    vision_model = load_model(model)
    print("---- {} loaded -----".format(model))
    labels = get_labels()
    img = "../data/butterfly.jpg"
    # makelogFile(model)

    start_time = datetime.datetime.now()
    print("Beginning inference")
    for i in range(5000):
        image = Image.open(img)
        # image.show()
        image_tensor = transform_image(image)
        label = predict(vision_model, image_tensor, labels)
        # print("My guess is {} \n\n".format(label))
    # stopNvidiaSmi()
    end_time = datetime.datetime.now()
    row = f"{model},{start_time},{end_time}"
    with open(pmfile, "a") as file1:
        file1.writelines(row)
    print("We're done with {}".format(model))


if __name__ == "__main__":
    main()
