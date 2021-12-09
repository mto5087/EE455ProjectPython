import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



'''
x = torch.tensor([5,3])
y = torch.tensor([3,4])
print(x*y)
x = torch.zeros([3,5])
print(x)
print(x.shape)
y = torch.rand([2,5])
print(y)
y = y.view([1,10])
print(y)
'''
start_time = time.time()

# Lets define a data set

train = datasets.MNIST("", train=True, download=False, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=False, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

for data in trainset:
	#print(data)
	break

x, y = data[0][0], data[1][0]
print(y)

#plt.imshow(data[0][0].view(28,28))
#plt.show()

#print(data[0][0].shape)

# How to confirm if the data is balanced
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
	Sx, ys = data
	for y in ys:
		counter_dict[int(y)] += 1
		total += 1
print(counter_dict)

#for i in counter_dict:
#	print(f"{i}:{counter_dict[i]/total*100}")

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 64) #self.fc1 = nn.Linear(input, output)
		# input = 28*28 = 784 from the images
		# output = the target we want 3 layers of 64 neurons so output is 64.
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, 10) # Here we have 10 classes for 0-10 as an output
		
	def forward(self, x): # This is how we want data to pass through
		x = F.relu(self.fc1(x)) # x passes through fully connected layers
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1) # dim 1 is equivalent to axes.  One of the outputs should basically become a 1.


net = Net()
print(net)

''''
X = torch.rand((28,28))
X = X.view(-1,28*28) # -1 specifies the input is an unknown shape.  You can also use 1.
output = net(X) # Now we passed data through the NN now we need to calculate how far off we were.
print(output)
'''

optimizer = optim.Adam(net.parameters(), lr = 0.001) # This is how we change the weights.  It is possible to freeze some weights if we want to but we do all of them here.  lr is the learning rate.  0.001 is the size of the step the optimizer will take.  So for the gradient this is the steps we will take to get to the bottom.  The smaller the step the longer it will take to learn, but the bigger the step you take a chance you will never get to the bottom.  If it is too small you might only find a local min instead of the global min.  You can come up with a varying learning rate that starts as big steps and then becomes smaller as you get to the bottom.  

# A full pass through the entire data set is called an epoch.  In this case we are going to pass through the entire data set 3 times.

EPOCHS = 1

for epoch in range (EPOCHS):
	for data in trainset:
		# Data is a batch of featuresets and labels
		X, y = data
		net.zero_grad()
		# There are two reasons to batch data.  It goes faster and there is a law of diminishing returns.  Batches help generalize.  This also works for weaker computers.  
		output = net(X.view(-1, 28*28))
		# now we want to calculate how wrong we were.
		loss = F.nll_loss(output, y)
		# now we want to backprop the loss
		loss.backward()
		optimizer.step() # This adjusts the weights for us
	print(loss)

correct = 0
total = 0
with torch.no_grad(): # We don't want to calculate the gradients during testing.
	for data in trainset:
		X, y = data
		output = net(X.view(-1, 784))
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Accuracy: ", round(correct/total, 3))
print("Run Time = ", time.time() - start_time)
for h in range(0,10):
    result = torch.argmax(net(X[h].view(-1, 784))[0])
    print(result)
    plt.imshow(X[h].view(28,28))
    plt.show()

#print(torch.argmax(net(X[0].view(-1, 784))[0]))
