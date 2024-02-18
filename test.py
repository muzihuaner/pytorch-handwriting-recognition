import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train, batch_size):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    device = next(net.parameters()).device
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 28*28))
            predictions = torch.argmax(outputs, dim=1)
            n_correct += torch.sum(predictions == y).item()
            n_total += y.size(0)
    return n_correct / n_total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = get_data_loader(is_train=True, batch_size=15)
    test_data = get_data_loader(is_train=False, batch_size=1)
    net = Net().to(device)
    
    print("初始准确率:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("第", epoch, "轮训练后的准确率:", evaluate(test_data, net))

    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)
        predict = torch.argmax(net.forward(x.view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x.cpu().view(28, 28))
        plt.title("Number: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
