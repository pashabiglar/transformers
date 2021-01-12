import torch
from torch import nn
import torch.nn.functional as F
import sys
torch.manual_seed(3)


class ffnn(nn.Module):
    def __init__(self):
        super(ffnn,self).__init__()
        self.layer1=    nn.Linear(10,100)
        self.layer2 = nn.Linear(100, 1000)
        self.layer3 = nn.Linear(1000, 4)

    def forward(self, data):
        data=self.layer1(data)
        data = F.relu(data)
        data=self.layer2(data)
        data = F.sigmoid(data)
        data = self.layer3(data)
        data = F.sigmoid(data)
        data=F.log_softmax(data)
        return data


def accuracy(list1,list2):
    total_right=0
    for x,y in zip(list1,list2):
        if(x==y):
            total_right+=1
    return total_right*100/len(list1)

gold_labels=torch.tensor(torch.randint(1,4,(10,1),dtype=torch.float),requires_grad=True)
model=ffnn()
input=torch.rand(10,10)
input.requires_grad=True
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=10)


def train(model):
    model.train()
    for epochs in range(10):
        optimizer.zero_grad()
        output=model(input)
        values, indices=output.topk(1)
        loss=nn.MSELoss()
        loss_output=loss(values,gold_labels)
        loss_output.backward()
        optimizer.step()
        scheduler.step(loss_output)
        print(f"epoch:{epochs} loss={loss_output}")
    return model

trained_model=train(model)
predicted_values=trained_model(input)
values, indices= predicted_values.topk(1)

print(f"accuracy={accuracy(indices,gold_labels)}")



