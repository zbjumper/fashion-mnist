
import torchvision
import torchvision.transforms as transforms
from matplotlib import pylab

data_dir = './data'
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
print("训练数据集条数", len(train_dataset))

val_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)
print("验证数据集条数", len(val_dataset))

im = train_dataset[0][0].numpy()

im = im.reshape(-1, 28)
pylab.imshow(im)

pylab.show()
print("该图片的标签为：", train_dataset[0][1])