"""
PyTorch [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

1. Working with data
2. Creating Models
3. Optimizing the Model Parameters
4. Saving Models
5. Loading Models

"""

import time
from typing import Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def prepare_data(batch_size: int = 64) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """准备 FashionMNIST 数据，返回 训练/验证 数据加载器

    Parameters
    ----------
    batch_size: each element in the dataloader iterable will return a batch of <batch_size> features and labels.
        default is 64

    Returns
    -------
    train_dataloader : data for train
    test_dataloader : data for test
    """
    print("Download data from open datasets ...")
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(),)

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(),)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    """深度学习使用的神经网络
    From `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. We define the layers of the network

    In the ``__init__`` function and specify how data will pass through the network in the ``forward`` function
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # type: ignore
            nn.ReLU(),  # 激活函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def create_model(device):
    return NeuralNetwork().to(device)


def train(dataloader, model, device, loss_fn, optimizer):
    """模型训练：在训练数据集上进行训练，并反向传播(backpropagates)失败信息以调整模型参数

    Parameters
    ----------
    dataloader: 训练数据集加载器
        每次返回指定 batch size 的数据
    model: 模型
    device: 运行模型的设备
    loss_fn: 损失函数 <https://pytorch.org/docs/1.10/nn.html#loss-functions>
    optimizer: 优化器 <https://pytorch.org/docs/1.10/optim.html>
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


######################################################################
# Loading Models
# ----------------------------
#
# The process for loading a model includes re-creating the model structure and loading
# the state dictionary into it.
def load_model_and_info():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    #############################################################
    # This model can now be used to make predictions.

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(),)
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    t0 = time.time()
    print("1. Prepare data ...")
    train_dataloader, test_dataloader = prepare_data()
    print(f"Took {time.time() - t0:.4f} seconds")
    print("2. Creating Models ...")
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"  # "mps" if torch.backends.mps.is_available() else
    print(f"Using {device} device")
    model = create_model(device)
    print(model)

    print("3. Optimizing the Model Parameters ...")
    t1 = time.time()

    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 优化算法: 调整学习律的策略


    ##############################################################################
    # The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
    # parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
    # accuracy increase and the loss decrease with every epoch.
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, device, loss_fn, optimizer)
        test(test_dataloader, model, device, loss_fn)
    print("Done!")
    print(f"Took {time.time() - t1:.4f} seconds")

    torch.save(model.state_dict(), "model.pth")
    print("4. Saving Models")
    print("Saved PyTorch Model State to model.pth")

    print("5. Loading Models: model.pth")
    load_model_and_info()
