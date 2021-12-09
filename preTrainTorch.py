from torchvision import models
from torchvision import transforms

import torch

resnet = models.densenet201(pretrained=True)
print(resnet)

transform = transforms.Compose([            #[1]
transforms.Resize(256),                    #[2]
transforms.CenterCrop(224),                #[3]
transforms.ToTensor(),                     #[4]
transforms.Normalize(                      #[5]
mean=[0.485, 0.456, 0.406],                #[6]
std=[0.229, 0.224, 0.225]                  #[7]
)])

from PIL import Image
img = Image.open("dog.jpeg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
print(out.shape)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(classes[index[0]], percentage[index[0]].item())
img.show()


