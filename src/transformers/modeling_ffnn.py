import torch
from torch import nn
import torch.nn.functional as F
import sys

class ffnn(nn.Module):
    def __init__(self):
        super(ffnn,self).__init__()
        self.layer1=    nn.Linear(10,10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 4)

    def forward(self,data):
        output1=self.layer1(data)
        output2=self.layer2(output1)
        output3 = self.layer3(output2)
        output4=F.log_softmax(output3)
        return output4


gold_labels=torch.randint(1,4,(10,1))
model=ffnn()
input=torch.rand(10,10)
input.requires_grad=True
output=model(input)
values, indices=output.topk(1)
print(gold_labels)
print(values)
loss=nn.MSELoss()
loss_output=loss(values,gold_labels)
print(loss_output)
loss_output.backward()


