from __future__ import print_function
import torch
from torch.autograd import Variable

x = torch.FloatTensor([[1,2],[3,4]])
var = Variable(x,requires_grad=True)
print(x)
print(var)

# careful var(a)*var(b) is dot multiply 
v_out = torch.mean(var*var)
print(v_out) #[[1,4],[9,16]] ==> 7.5
print(type(v_out))
v_out.backward()   # backpropagation from v_out => x

# x^2/4 =>  x/2 is var gradient
print(var.grad)   

x = Variable(torch.FloatTensor([1]),requires_grad=True)
y = x.pow(2) # y=x^2
z = 2*y + 3
z.backward()
print(x.grad)  # 4
# None because y have father node ,backward() is calculate leaf node
print(y.grad)  

x = Variable(torch.FloatTensor([1]),requires_grad=True)
y = Variable(torch.FloatTensor([2]),requires_grad=True)
z = x.pow(2) + 3*y
z.backward()
print(x.grad) # grad is 2*x =>2
print(y.grad) # grad is 3

