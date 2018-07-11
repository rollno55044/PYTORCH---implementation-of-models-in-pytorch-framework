import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# Hyper-parameters
input_size = 784

hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./' ,train=True ,download=True,transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root='./',
                                           train = False,
                                          transform = transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader (dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False)

test_loader = torch.utils.data.DataLoader (dataset= test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self,input_size ,hidden_size ,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model =  NeuralNet(input_size,hidden_size,num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() ,lr=learning_rate)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
# Move tensors to the configured device
       images = images.reshape(-1 ,28 * 28).to(device)
       labels = labels.to(device)

# forward pass
       output = model(images)
       loss  = criterion(output,labels)

# Backward and optimiz
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


       if (i+1) % 100 == 0 :
            print('Epoch [{}/{}] ,step [{}/{}] , Loss:{:.4f}'
                  .format(epoch+1 , num_epochs, i+1 , total_step , loss.item()))


 # Test the model
# In test phase, we don't need to compute grad (for memory efficiency)
with torch.no_grad():
    correct =  0
    total = 0

    for images , labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')









