from __future__ import print_function
import torch
import torch.utils.data as Data

# create fake data
data_tensor = torch.randn(5,3)
target_tensor = torch.randn(5)
# class torch.utils.data.TensorDataset(data_tensor, target_tensor)
tensor_dataset = Data.TensorDataset(data_tensor,target_tensor)

print(
	'\ndata_tensor:',data_tensor,
	'\ntarget_tensor:',target_tensor,
)
# using index 
print(
	'\ntensordata[0]:',tensor_dataset[0],
)
print('len of tensor_dataset',len(tensor_dataset))

###### Dataloader encapsulation Dataset or subclass as a iterater
"""
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
"""
tensor_dataloader = Data.DataLoader(tensor_dataset,batch_size=2,shuffle=True,num_workers=2)
print("************ iterater the dataset **********")
for data,target in tensor_dataloader:
		print(data,target)
print('a batch tensor data',iter(tensor_dataloader).next())
