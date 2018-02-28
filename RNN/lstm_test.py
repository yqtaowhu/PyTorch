"""
this is a test of lstm
we should know every parameters shape
author : taoyanqi
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
rnn = nn.LSTM(
	input_size = 10,
	hidden_size = 20,
	num_layers = 2,
)

input_1 = Variable(torch.randn(5,3,10)) #(seq_len, batch, input_size)
#h0 (num_layers * num_directions, batch, hidden_size)
h0 = Variable(torch.randn(2,3,20)) 
c0 = Variable(torch.randn(2,3,20))

out,(hn,hc) = rnn(input_1,(h0,c0))
print(out.size()) # out is (seq_len,batch,hidden_size)

for line in out[-1,:,:]:  # the last output (batch_size,hidden_size)
	print(line.data.numpy()) # Variable => numpy
