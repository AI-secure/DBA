import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class MnistNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(MnistNet, self).__init__(f'{name}_Simple', created_time)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.fc2 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # in_features = 28 * 28
        # x = x.view(-1, in_features)
        # x = self.fc2(x)

        # normal return:
        return F.log_softmax(x, dim=1)
        # soft max is used for generate SDT data
        # return F.softmax(x, dim=1)

if __name__ == '__main__':
    model=MnistNet()
    print(model)

    # import numpy as np
    # from torchvision import datasets, transforms
    # import torch
    # import torch.utils.data
    # import copy
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #
    # train_dataset = datasets.MNIST('./data', train=True, download=True,
    #                                     transform=transforms.Compose([
    #                                         transforms.ToTensor(),
    #                                         # transforms.Normalize((0.1307,), (0.3081,))
    #                                     ]))
    # test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.1307,), (0.3081,))
    # ]))
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                           batch_size=64,
    #                                           shuffle=False)
    # client_grad = []
    #
    # for batch_id, batch in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     data, targets = batch
    #     output = model(data)
    #     loss = nn.functional.cross_entropy(output, targets)
    #     loss.backward()
    #     for i, (name, params) in enumerate(model.named_parameters()):
    #         if params.requires_grad:
    #             if batch_id == 0:
    #                 client_grad.append(params.grad.clone())
    #             else:
    #                 client_grad[i] += params.grad.clone()
    #     optimizer.step()
    #     if batch_id==2:
    #         break
    #
    # print(client_grad[-2].cpu().data.numpy().shape)
    # print(np.array(client_grad[-2].cpu().data.numpy().shape))
    # grad_len = np.array(client_grad[-2].cpu().data.numpy().shape).prod()
    # print(grad_len)
    # memory = np.zeros((1, grad_len))
    # grads = np.zeros((1, grad_len))
    # grads[0] = np.reshape(client_grad[-2].cpu().data.numpy(), (grad_len))
    # print(grads)
    # print(grads[0].shape)


