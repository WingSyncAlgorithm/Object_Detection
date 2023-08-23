# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:47:18 2023

@author: DavidChou23
https://hackmd.io/@Maxlight/SkuYB0w6_
https://www.youtube.com/watch?v=OP5HcXJg2Aw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=10
"""
#import modules
import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True  #true or false to download mnist data

#mnist
train_data = torchvision.datasets.MNIST(root = './mnist',train = True,transform = torchvision.transforms.ToTensor(),download = DOWNLOAD_MNIST)
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.ion()

"""
#show data
for i in range(11):
  plt.imshow(train_data.train_data[i].numpy(), cmap = 'gray')
  plt.title('%i' % train_data.train_labels[i])
  plt.pause(0.5)
plt.show()
"""

#data loader(iterator)
train_loader = data_.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True,num_workers = 4)
test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)

#class inheritance from nn.module
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.out = nn.Linear(32*7*7, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output, x

#module
cnn = CNN()
print(cnn) #print module structure

optimization = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

#just for graph
epoch_l=[i for i in range(1,EPOCH+1,1)]
accuracy_l=[]
loss_l=[]


#Training
if __name__=='__main__': #to prevent dataloader's threading Error
    train_x = torch.unsqueeze(test_data.test_data, dim = 1).type(torch.FloatTensor)[:6000]/255.
    train_y = test_data.test_labels[:6000]

    test_x =  torch.unsqueeze(test_data.test_data, dim = 1).type(torch.FloatTensor)[6000:]/255.
    test_y = test_data.test_labels[6000:]

    for epoch in range(EPOCH):
      print("epoch: ",epoch)
      for step, (batch_x, batch_y) in enumerate(train_loader):
        bx = Variable(batch_x)
        by = Variable(batch_y)
        output = cnn(bx)[0]
        loss = loss_func(output, by)
        optimization.zero_grad()
        loss.backward()
        optimization.step()
    
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
      accuracy_l.append(float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0)))
      loss_l.append(loss.data.numpy())
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

#save model
torch.save(cnn.state_dict(), "./cnn")
#export variable
import pickle
pickle.dump(accuracy_l, open("accuracy_l.dat",'wb'))
pickle.dump(loss_l, open("loss_l.dat",'wb'))

#loads model
"""
cnn1=CNN()
cnn1.load_state_dict(torch.load("./cnn"))
cnn1.eval()
#load variable
import pickle
accuracy_l=pickle.load(open("./accuracy_l.dat",'rb'))
loss_l=pickle.load(open("./loss_l.dat",'rb'))
"""

#count accuracy of dataset with model
def accuracy_loss(model,data,label):
    out,_=model(data)
    pred_y=torch.max(out,1)[1].data.numpy()
    accu_y=label.numpy()
    tmp=[pred_y[i]==accu_y[i] for i in range(0,len(pred_y),1)]
    return tmp.count(True)/len(tmp)*100, tmp.count(False)/len(tmp)

#generate matrix
def confusion_matrix(model,data,label):
    out,_=model(data)
    pred_y=torch.max(out,1)[1].data.numpy()
    accu_y=label.numpy()
    w, h = 10, 10
    matrix = [[0 for x in range(w)] for y in range(h)]
    tmp=[(pred_y[i],accu_y[i]) for i in range(0,len(pred_y),1)]
    for item in tmp:
        matrix[item[0]][item[1]]+=1
    return matrix

#show matrix in graph
def show_confusion_matrix(mat):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    # using the matshow() function
    caxes = axes.matshow(mat)
    figure.colorbar(caxes)
    figure.show()


#plot image
plt.plot(epoch_l,list(map(float,accuracy_l)),'--')
plt.xlim([])
plt.plot(epoch_l,list(map(float,loss_l)),'--')
plt.xlabel("$epoch$")
plt.ylabel("$percentage$")
plt.legend(["learning","loss"])
plt.title("Learning curve& loss curve")
plt.show()

#plot confusion matrix in graph
matrix=confusion_matrix(cnn, test_x, test_y)
show_confusion_matrix(matrix)
