from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

class mnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.stacks=nn.Sequential(
            nn.Conv2d(1,64,5),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU(),
        )
        
    def forward(self,x):
        y=self.stacks(x)
        output = F.log_softmax(y, dim=1)

        return output
        
    
def train(module:nn.Module,trainDatas:torch.utils.data.DataLoader,lossFunc:nn.CrossEntropyLoss(),optimiz:optim.SGD,epoch:int):
    module.train()
    for batch,(data,target) in enumerate(trainDatas):
        data,target = data.to("cpu"),target.to("cpu")
        output = module(data)
        optimiz.zero_grad()
        # loss = lossFunc(output,target)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimiz.step()
        if batch % 100 ==0:
            print(f'Train Epoch: {epoch} [{batch * len(data)}/{len(trainDatas.dataset)} ({(100*batch/len(trainDatas)):>.2f}%)]\tLoss: {loss.item():>.5f}')
            
def test(module:nn.Module,testDatas:torch.utils.data.DataLoader,lossFunc:nn.CrossEntropyLoss):
    module.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for data,target in testDatas:
            # data = data.to("cpu")
            # target = target.to("cpu")
            pred = module(data)
            loss+=lossFunc(pred,target).item()
            loss = F.nll_loss(pred,target)

            acc+=  (pred.argmax(1) == target).type(torch.float).sum().item()
    loss /= len(testDatas)#
    acc /= len(testDatas.dataset)
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {loss:>8f} \n")

        
        

def main():
    trainDataset = datasets.MNIST("data",download=True,train=True,transform=transforms.ToTensor())
    testDataset = datasets.MNIST("data",download=True,train=False,transform=transforms.ToTensor())
    
    #load data with DataLoader
    trainData = torch.utils.data.DataLoader(batch_size=64,shuffle=True,dataset=trainDataset)
    testData = torch.utils.data.DataLoader(batch_size=100,shuffle=True,dataset=testDataset)
    
    learningRate = 10
    epoch = 5
    
    module = mnistNet().to("cpu")
    optimizer = optim.SGD(module.parameters(),lr=learningRate)
    lossFN = nn.CrossEntropyLoss()
    
    for i in range(epoch):
        train(module=module,optimiz=optimizer,lossFunc=lossFN,epoch=i,trainDatas=trainData)
        test(module=module,testDatas=testData,lossFunc=lossFN)
    

    
if __name__ == "__main__":
    main()
    
    
    
    