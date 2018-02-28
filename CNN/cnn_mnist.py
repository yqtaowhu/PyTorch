from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_MNIST = False

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=5,
				stride=1,
				padding=2,
			),                  #shape(16*28*28)
			nn.ReLU(),      
			nn.MaxPool2d(kernel_size=2),  #(16*14*14)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16,32,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),  #(32*7*7)
		)
		self.out = nn.Linear(32*7*7,10)   #full connect layer
	# build fowrd layer
	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0),-1) # flatten (batch_size,32*7*7)
		output = self.out(x)
		return output   # shape : (batch_size,10)

cnn = CNN()
print(cnn)

# torchvision.datasets is subclass torch.data.Dataset
train_data = torchvision.datasets.MNIST(
	root = '../mnist',
	train = True,
	transform = torchvision.transforms.ToTensor(), # transform dataset 
	download = DOWNLOAD_MNIST  
)
#print(train_data.train_data.size())
#print(train_data.train_labels.size())
#plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
#plt.title('label is %i' % train_data.train_labels[0])
#plt.show()


# data loader is subclass dataset
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,
	shuffle=True,num_workers=2)
# we can using iterater to take value of train_loader
#print("************************ iterater train_loader ****************")
#print(iter(train_loader).next()) # a batch
test_data = torchvision.datasets.MNIST(root='../mnist',train=False) # is Dataset class
# test_data is tuple (data,target)
print(test_data.test_data.size()) #(10000,28,28)

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(
	torch.FloatTensor)[:2000]/255. # (2000,28,28) => (2000,1,28,28)
test_y = test_data.test_labels[:2000]

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
	for step,(data,target) in enumerate(train_loader):
		batch_data = Variable(data)
		batch_target = Variable(target)
		
		output = cnn(batch_data) # forward
		loss = loss_func(output,batch_target)
		optimizer.zero_grad()   # clear gradients for this training step
		loss.backward()
		optimizer.step()        # apply graients
		# test
		if step % 100 == 0:
			test_out = cnn(test_x)  # 2000*10
			# torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
			pred_y = torch.max(test_out,1)[1].data.squeeze() # 2000*1 => 2000 vector
			accuracy = sum(pred_y == test_y)/float(test_y.size(0)) # tensor
			print('Epoch:',epoch,'step:',step,'loss:%.4f' % loss.data[0],
				'test accuracy %.2f' % accuracy)
			torch.save(cnn,'cnn.pkl')

net = torch.load('cnn.pkl')
test = net(test_x[:10])
pred_y = torch.max(test,1)[1].data.numpy().squeeze()
print(
	'\nreal number:' , test_y[:10].numpy(),   # tensor => numpy
	'\nprediction number:' , pred_y,
)


