import torch
import torchvision
import torchvision.transforms as transforms

from model import myConNet

data_dir = './data'
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(
    data_dir, train=True, transform=transform, download=True)
val_dataset = torchvision.datasets.FashionMNIST(
    data_dir, train=False, transform=transform, download=True)

batch_size = 10

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False)

net = myConNet()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epoches = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(epoches):
    running_loss = 0.0
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d,%5d] loss:%.3f' % (epoch+1, i+1, running_loss/1000))
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), './model/net.pth')