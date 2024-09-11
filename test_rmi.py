
import torch
from rmi import RMILoss

loss = RMILoss(with_logits=True)
#batch_size, classes, height, width = 5, 4, 64, 64 
batch_size, classes, height, width = 5, 4, 1, 1

pred = torch.randn(batch_size, classes, height, width, requires_grad=True)
target = torch.randn(batch_size, classes, height, width).random_(2) 

output = loss(pred, target)
output.backward()
