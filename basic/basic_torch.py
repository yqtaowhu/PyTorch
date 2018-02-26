# more about you can refrence pytorch.org/docs/0.3.0/torch.html

from __future__ import print_function
import torch 
import numpy as np
#######################  how to create a tensor
# 1. numpy => tensor
np_data = np.arange(6).reshape((2,3))
# ndarray => tensor
tensor = torch.from_numpy(np_data)
# tensor => ndarray
tensor2array = tensor.numpy()
print(
	'\nnumpy:',np_data,
	'\ntensor:',tensor,
	'\ntensor2array:',tensor2array,
	'\n'
)
# 2. list => tensor
data = [1,2,3]
tensor = torch.FloatTensor(data)
print(tensor)
# 3. use function 
print(torch.eye(3))
print(torch.linspace(-5,5,10)) # (start,end,step)
print(torch.ones(2,3)) # 2*3 matrix
######################## basic usage
# judge obj is tensor or not
print(torch.is_tensor(tensor)) #True
# torch.numel(input) => int,return number of element of a tensor
a = torch.zeros(4,4)
print(torch.numel(a)) # 16

# Indexing,Slicing,etc
# torch.cat(seq,dim=0) => tensor,0 is by rows(usualy)
print("***********usage of cat**********")
x = torch.randn(2,3)  # create a tensor 2*3
print(torch.cat((x,x,x),0))
print(torch.cat((x,x,x),1))

# torch.squeeze(input,dim=None,out=None)
# remove 1 dim like [[[2,1],[1,2]]] dims:1*2*2 => 2*2
print("************usage of squeeze********")
x = torch.zeros(2,1,2,2,1)
print(x.size())
y = torch.squeeze(x)
print(y.size())
