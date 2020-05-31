import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import urllib.request
import json
import argparse


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def main(img="data/butterfly.jpg", model="vgg19"):
    vision_model = load_model(model)
    if vision_model == None:
        print("Check that model name")
        return
    print("________ Model ready ____________")
    image = Image.open(img)
    image.show()
    image_tensor = transform_image(image)
    print("Image dimension: {}".format(image_tensor.size()))
    print("_________________________________")
    labels = get_labels()
    print("__________Labels loaded__________\n\n")
    label = predict(vision_model, image_tensor, labels)
    print("My guess is {} \n\n".format(label))


def load_model(model_name):
    try:
        model = models.__dict__[model_name](pretrained=True).to(device)
        return model
    except:
        print('Can\'t fetch {}'.format(model_name))


def transform_image(image):
    max_size = 224

    if max_size < max(image.size):
        size = max_size
    else:
        size = max(image.size)

    transformation = transforms.Compose([
        torchvision.transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image)  # shape [3,224,224]
    image_tensor = image_tensor.unsqueeze(0)  # shape [1,3,224,224]
    return image_tensor


def get_labels():
    with urllib.request.urlopen("https://raw.githubusercontent.com/shivangidas/image-classifier/master/mobilenet/imagenet_classes.json") as url:
        data = json.loads(url.read().decode())
        return data


def predict(vgg, image_tensor, labels):
    input_image = image_tensor.to(device)
    outputs = vgg(input_image).detach().to(device)
    highest_pred = torch.argmax(outputs)
    tada = labels[str(highest_pred.item())]
    return tada


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--path', type=str,
                        help='path to an image')
    parser.add_argument('--model', type=str,
                        help='vision model to be used')
    args = parser.parse_args()

    if args.path and args.model:
        main(args.path, args.model)
    elif args.path:
        main(path=args.path)
    elif args.model:
        main(model=args.model)
    else:
        main()
