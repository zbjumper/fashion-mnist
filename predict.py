import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import myConNet
data_dir = './data'
batch_size = 10

transform = transforms.Compose([transforms.ToTensor()])

val_dataset = torchvision.datasets.FashionMNIST(
    data_dir, train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


network = myConNet()

network.load_state_dict(torch.load('./model/net.pth'))

network.to(device)
dataiter = iter(test_loader)
images, labels = next(dataiter)

inputs, labels = images.to(device), labels.to(device)

imshow(torchvision.utils.make_grid(images, nrow=batch_size))


print('真实标签:     ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

outputs = network(inputs)
# print(outputs.shape)
_, predicted = torch.max(outputs, 1)
# print(_)
# print('predicted shape:', predicted.shape)
# print('predicted:', predicted)

print('真实预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(images))))

#测试模型
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        inputs, labels = images.to(device), labels.to(device)
        outputs = network(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.to(device)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

sumacc = 0
for i in range(10):
    Accuracy = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy ))
    sumacc =sumacc+Accuracy
print('Accuracy of all : %2d %%' % ( sumacc/10. ))