# 01001101 01100001 01110011 01110100 01111001 01000100 01100101 01110110
from colors import colors as s
from time import process_time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
t_start=process_time()
# Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load Data
train_dataset = dataset.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)
print(s.fg.yellow,"Neural Network Initialized",s.reset)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
print(s.fg.lightcyan,f"Training data...{s.reset} batch size:{s.bold}{batch_size}{s.reset} epochs:{s.bold}{num_epochs}{s.reset}")
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Get the correct shape
        data = data.reshape(data.shape[0], -1)
        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()

# wall_time
t_stop=process_time()

# Check accuracy on training & test to see how good our model is
def check_accuuracy(loader, model):
    if loader.dataset.train:
        print(s.bold,"Checking accuracy on training data",s.reset)
    else:
        print(s.bold,"Checking accuracy on test data",s.reset)
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples +=  predictions.size(0)
        print(s.fg.lightgrey,f"Got {num_correct}/{num_samples} with accuracy of {s.fg.green}{s.bold}{float(num_correct)/float(num_samples)*100:.2f}%",s.reset)
    model.train()
check_accuuracy(train_loader, model)
check_accuuracy(test_loader, model)

wall_time = t_stop - t_start
print(s.bold,f"Elapsed time: {s.reset}{s.fg.yellow}{wall_time:.2f} sec{s.reset}")