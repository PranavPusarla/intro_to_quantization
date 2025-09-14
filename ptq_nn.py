import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torchao.utils import get_model_size_in_bytes
import argparse
import os
import copy

batch_size = 64
learning_rate = 1e-4
epochs = 200

torch.backends.quantized.engine = "qnnpack"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="Post-Training Quantization (INT8 dynamic) for a small NN")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the model to the models/ptq_nn directory")
    parser.add_argument("--load_model", type=bool, default=False, help="Load the model from the models/ptq_nn directory")
    args = parser.parse_args()

    model_fp32 = Network()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_fp32.parameters(), lr=learning_rate)

    if args.load_model:
        model_fp32.load_state_dict(torch.load(f"models/ptq_nn/fp32.pt"))
    else:
        for epoch in range(epochs):
            model_fp32.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model_fp32(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model_fp32.eval()

    fp32_correct = 0
    fp32_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model_fp32(data)
            _, predicted = torch.max(output.data, 1)
            fp32_total += target.size(0)
            fp32_correct += (predicted == target).sum().item()

    model_int8 = copy.deepcopy(model_fp32)
    quantize_(model_int8, Int8WeightOnlyConfig())

    model_int8.eval()

    int8_correct = 0
    int8_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model_int8(data)
            _, predicted = torch.max(output.data, 1)
            int8_total += target.size(0)
            int8_correct += (predicted == target).sum().item()

    if args.save_model:
        os.makedirs("models/ptq_nn", exist_ok=True)
        torch.save(model_fp32.state_dict(), f"models/ptq_nn/fp32.pt")

    print("\n-------BASELINE METRICS--------")
    print(f"Test Accuracy FP32: {100 * fp32_correct / fp32_total:.2f}%")
    print(f"FP32 SIZE: {get_model_size_in_bytes(model_fp32) / 1e6:.2f} MB")
    print("--------------------------------\n")
    print("-------QUANTIZED METRICS---------")
    print(f"Test Accuracy INT8: {100 * int8_correct / int8_total:.2f}%")
    print(f"INT8 SIZE: {get_model_size_in_bytes(model_int8) / 1e6:.2f} MB")
    print("--------------------------------\n")

if __name__ == "__main__":
    main()