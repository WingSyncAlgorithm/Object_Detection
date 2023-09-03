#import modules
import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
from tqdm.auto import tqdm
import pickle
from google.colab import files
#class inheritance from nn.module
class CNN(nn.Module):  #inheritance from nn.Module
  def __init__(self):
    super(CNN,self).__init__()
    self.conv1=nn.Conv2d(1,16,5,1,2)
    self.act1=nn.ReLU()
    self.pool1=nn.MaxPool2d(2)
    self.conv2=nn.Conv2d(16,32,5,1,2)
    self.act2=nn.ReLU()
    self.pool2=nn.MaxPool2d(2)
    self.fc1=nn.Linear(32*7*7,10) #fullconnected

  def forward(self,x):
    x=self.conv1(x)
    x=self.act1(x)
    x=self.pool1(x)

    x=self.conv2(x)
    x=self.act2(x)
    x=self.pool2(x)

    x=x.view(x.size(0),-1)
    output=self.fc1(x)

    return output,x

def MNIST_datasets(root,download):
  train_data = torchvision.datasets.MNIST(root = root,
                    train = True,  #load data from training.pt
                    transform = torchvision.transforms.ToTensor(),  #轉成一維tensor
                    download = DOWNLOAD_MNIST)
  test_data = torchvision.datasets.MNIST(root = root,
                    train = False,  #load data from test.pt
                    download=DOWNLOAD_MNIST,
                    transform=torchvision.transforms.ToTensor())
  return train_data, test_data

def loader(dataset,batch,num_workers,device,shuffle=True):
  train_loader = data_.DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=num_workers,pin_memory=(device=='cuda'))  #用幾個process去load data到ram
  return train_loader

def training(model,device,EPOCH,batch,Learning_rate,train_loader,test_loader):
  model.to(device)
  optimization=torch.optim.Adam(model.parameters(),Learning_rate)
  loss_func = nn.CrossEntropyLoss()
  t_accuracy_l,t_loss_l=[],[]
  v_ac_l,v_loss_l=[],[]
  w, h = 10, 10
  matrix = [[0 for x in range(w)] for y in range(h)]
  for epoch in tqdm(range(EPOCH),desc='epoch'):
    model.train()
    #print("epoch: ",epoch)
    #training
    train_loss=0
    train_correct=0
    for (batch_x,batch_y) in tqdm(train_loader,desc='epoch='+str(epoch)+" ",total=len(train_loader)):
      bx= Variable(batch_x).to(device)
      by= Variable(batch_y).to(device)
      out=model(bx)[0]
      loss=loss_func(out,by)
      optimization.zero_grad()
      loss.backward()
      optimization.step()
      train_loss+=loss.item()
      predict=torch.max(out,1)[1].cpu().data.numpy()
      train_correct += float((predict == by.cpu().data.numpy()).astype(int).sum()) / float(by.size(0))
    t_accuracy_l.append(train_loss/len(train_loader))
    t_loss_l.append(train_correct/len(train_loader))

    #record accurracy
    model.eval()
    test_loss=0
    test_correct=0
    for datas,labels in test_loader:
      datas=Variable(datas).to(device)
      labels=Variable(labels)
      outputs = model(datas)[0]
      loss = loss_func(outputs, labels.to(device))
      test_loss += loss.item()
      predicted = torch.max(outputs, 1)[1].cpu().data.numpy()
      test_correct += float((predicted == labels.data.numpy()).astype(int).sum())/float(labels.size(0))
      if (epoch == (EPOCH-1)):
        tmp=[(predicted[i],labels.data.numpy()[i]) for i in range(0,len(predicted),1)]
        for item in tmp:
          matrix[item[0]][item[1]]+=1
    v_loss_l.append(test_loss/len(train_loader))
    v_ac_l.append(test_correct/len(test_loader))


    """
    test_x = (torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)[:]/255.0).to(device)
    test_y = test_data.targets[:]
    test_output = cnn(test_x)[0]
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    accuracy_l.append(float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0)))
    loss_l.append(loss.cpu().data.numpy())
    """
  torch.cuda.empty_cache()
  return t_accuracy_l,t_loss_l,v_ac_l,v_loss_l,matrix
def draw_curve(EPOCH,t_accuracy_l,t_loss_l,v_ac_l,v_loss_l):
  plt.figure()
  plt.subplot(1,2,1)
  plt.plot(range(EPOCH),t_accuracy_l,'x-')
  plt.plot(range(EPOCH),v_ac_l,'o-')
  plt.legend(['training_accuracy','testing_accuracy'])
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.title('learning_curve')
  plt.subplot(1,2,2)
  plt.plot(range(EPOCH),t_loss_l,'-x')
  plt.plot(range(EPOCH),v_loss_l,'-o')
  plt.legend(['training_loss','testing_loss'])
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('loss_curve')
  plt.show()

def draw_matrix(mat):
  #show confusion matrix
  figure = plt.figure()
  axes = figure.add_subplot(111)
  # using the matshow() function
  caxes = axes.matshow(mat)
  figure.colorbar(caxes)
  figure.show()


if __name__=='__main__': #to prevent dataloader's threading Error
  #select a mode
  mode='train' #train/load
  #mode='load'
  print("The mode running is :"+mode)

  torch.cuda.empty_cache()
  #preset
  root='./minst/'
  DOWNLOAD_MNIST = True  #true or false to download mnist data
  num_worker =0
  EPOCH = 20
  BATCH_SIZE = 50
  LR = 0.001
  draw_graph=True
  model_path='./cnn_ep'+str(EPOCH)+'.pt'
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)

  train_data,test_data = MNIST_datasets(root,DOWNLOAD_MNIST)
  train_loader = loader(train_data,BATCH_SIZE,num_worker,device,True)
  valid_loader = loader(test_data,BATCH_SIZE,num_worker,device,True)
  
  #main
  if mode=='train':
    cnn=CNN()
    t_accuracy_l,t_loss_l,v_ac_l,v_loss_l,matrix=training(cnn,device,EPOCH,BATCH_SIZE,LR,train_loader,valid_loader)
    pickle.dump(t_accuracy_l,open("./result/accuracy.bin",'wb'))
    pickle.dump(t_loss_l,open("./result/loss.bin",'wb'))
    pickle.dump(v_ac_l,open("./result/v_ac.bin",'wb'))
    pickle.dump(v_loss_l,open("./result/v_loss.bin",'wb'))
    torch.save(cnn, './result/+'model_path)
    #!zip './colab_training.zip' './result'
    #files.download('./colab_traing.zip')
  elif mode=='load':
    #check file exist-->(upload?)-->load
    cnn     =torch.load(open(model_path,'rb'))
    cnn.eval()

    t_accuracy_l = pickle.load(open("accuracy.bin",'rb'))
    t_loss_l   = pickle.load(open("loss.bin",'rb'))
    v_ac_l    = pickle.load(open("v_ac.bin",'rb'))
    v_loss_l   = pickle.load(open("v_loss.bin",'rb'))

  if draw_graph:
    draw_curve(EPOCH,t_accuracy_l,t_loss_l,v_ac_l,v_loss_l)
    draw_matrix(matrix)

"""
#test model
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
"""
