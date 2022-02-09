from PIL import Image
import torch
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)
# should be 1.8.0

def load_model():
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    return model

def run(model):
    
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    img = Image.open('./data')
    img = transform(img)[None,]
    out = model(img)
    clsidx = torch.argmax(out)
    return clsidx.item()