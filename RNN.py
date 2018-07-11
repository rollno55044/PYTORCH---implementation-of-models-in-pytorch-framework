import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.009

#MNIST DATASET

train_dataset = torchvision.datasets.MNIST(root = './' ,
                                           train = True ,
                                           transform = transforms.ToTensor(),
                                           download= True)
test_dataset = torchvision.datasets.MNIST(root = './',
                                          train = False ,
                                          transform = transforms.ToTensor())
# data loader
train_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# rnn
class RNN(nn.Module):
    def __init__(self,input_size , hidden_size, num_layers , num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)


    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #  forward lstm
        out, _ = self.lstm(x , (h0, c0)) #  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # decode the state of last time
        out = self.fc(out [:, -1 ,:])

        return out

model = RNN(input_size, hidden_size , num_layers , num_classes).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i ,(images, labels) in enumerate(train_loader):
        images = images.reshape(-1,sequence_length ,input_size).to(device)
        labels = labels.to(device)


        # forwarf pass

        output = model(images)
        loss = criterion(output , labels)


        # backward and optimize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images ,labels in test_loader:
        images = images.reshape(-1 , sequence_length , input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# save model

torch.save(model.state_dict() , 'model.ckpt')