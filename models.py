import torch
import torchvision
from torchstat import stat
import torchvision.models as models
from thop import profile
import os

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")  # and "inception" in name
                     and callable(models.__dict__[name]))


def get_vision_models():
    model_list = torch.hub.list('pytorch/vision', force_reload=False)
    print("Number of vision models: {}".format(len(model_list)))
    return model_list


def create_folder(folder_name):
    path = os.getcwd()
    print("The current working directory is %s" % path)
    path = path + '/' + folder_name
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def get_flops(model, folder_name, name):
    path = os.getcwd()
    path = path + '/' + folder_name
    #stat(model, (3, 224, 224))
    with open(path + "/models.txt", "a") as file1:
        # Writing data to a file
        dsize = (1, 3, 224, 224)
        if "inception" in name:
            dsize = (1, 3, 299, 299)
        inputs = torch.randn(dsize).to(device)
        macs, params = profile(model, (inputs,), verbose=False)
        #print(macs, params)
        row = name + "," + str(macs) + "," + str(params) + "\n"
        file1.writelines(row)


if __name__ == "__main__":
    #vision_model_list = get_vision_models()
    create_folder('logs')
    # model = torch.hub.load(
    #     'pytorch/vision', vision_model_list[0], pretrained=True)
    for name in model_names:
        try:
            model = models.__dict__[name]().to(device)
        except:
            print('Can\'t fetch {}'.format(name))
        get_flops(model, 'logs', name)
