import sys
in_colab = ('google.colab' in sys.modules)
import os
from datetime import datetime
if(in_colab):
  from google.colab import files #for downloading file
  from tqdm.auto import tqdm #process bar
else:
  import tqdm

import torch #pytorch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn #neural network
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

#load dataset
def MNIST_datasets(root,download):
  """
  download mnist data
  root: where to store the data be download
  download: true or false to download data
  trainsforms data to torch tensor
  return train_dataset, test_dataset
  """
  # transforms.ToTensor():將圖像數據從PIL圖像（Pillow圖像庫的格式）轉換為PyTorch的張量（tensor）格式
  train_data = torchvision.datasets.MNIST(root = root,
                    train = True,  #load data from training.pt
                    transform = torchvision.transforms.ToTensor(),  #轉成一維tensor
                    download = download)
  test_data = torchvision.datasets.MNIST(root = root,
                    train = False,  #load data from test.pt
                    download=download,
                    transform=torchvision.transforms.ToTensor())
  return train_data, test_data

#create dataloader

def loader(dataset,batch,num_workers,device,shuffle=True):
  """
  https://hackmd.io/3XMJi64dT_aJ8VFSNJ_hJw
  return dataloader of dataset
  """
  train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=num_workers,pin_memory=(device=='cuda'))  #用幾個process去load data到ram
  return train_loader

#define cnn module
#class inheritance from nn.module
class CNN(nn.Module):  #inheritance from nn.Module
  def __init__(self):  #constructor
    super(CNN,self).__init__()  # call nn.Moudle's constructor
    self.conv1=nn.Conv2d(1,16,5,1,2)  #convolution layer, in channel=1(gray), out channel=16, kernal_size=5*5, stride=1, padding=2(add 2 layer of "0")
    self.act1=nn.ReLU()  #ReLU activation function
    self.pool1=nn.MaxPool2d(2)  #max pooling, kernal size=2
    self.conv2=nn.Conv2d(16,32,5,1,2)  #convolution layer
    self.act2=nn.ReLU()  #ReLU activation function
    self.pool2=nn.MaxPool2d(2)  # max pooling
    self.fc1=nn.Linear(32*7*7,10) #fully connected layer, 32(32 channel),7*7(data size after pooling)

  def forward(self,x): #how data pass from different layer
    x=self.conv1(x)
    x=self.act1(x)
    x=self.pool1(x)

    x=self.conv2(x)
    x=self.act2(x)
    x=self.pool2(x)

    x=x.view(x.size(0),-1)  #flatten to 1d
    output=self.fc1(x)  #fully connected layer

    return output,x


def training(model,device,EPOCH,batch,Learning_rate,train_loader,test_loader):
  model.to(device) #turn model mode to torch.device(cpu or cuda gpu)
  optimization=torch.optim.Adam(model.parameters(),Learning_rate) #use Adam as optimization function
  loss_func = nn.CrossEntropyLoss() #use crossEntropyLoss to calculate loss as loss function
  #what is crossEntropy --> https://hackmd.io/dI-J6ApjSiWhbAt2wrqCDA
  w, h = 10, 10 #for matrix dimension use
  history={'train_acc':[],'train_loss':[],'valid_acc':[],'valid_loss':[],'matrix': [[0 for x in range(w)] for y in range(h)]} #for recording
  for epoch in tqdm(range(EPOCH),desc='epoch'):#tqdm is for procession bar
    model.train() #turn model to train mode
    #print("epoch: ",epoch)
    #training
    train_loss=0
    train_correct=0
    for (batch_x,batch_y) in tqdm(train_loader,desc='epoch='+str(epoch)+" ",total=len(train_loader)):
      bx= Variable(batch_x).to(device)
      by= Variable(batch_y).to(device)
      out=model(bx)[0] #get output probibility array
      loss=loss_func(out,by) #loss function
      optimization.zero_grad() #clear gradient
      loss.backward() #backward propogation
      optimization.step() #adjust parameter

      #calulate loss, accurracy
      predict=torch.max(out,1)[1].cpu().data.numpy()
      #print((float((predict != by.cpu().data.numpy()).astype(int).sum()) / float(by.size(0))),(float((predict == by.cpu().data.numpy()).astype(int).sum()) / float(by.size(0))))
      #train_loss+=(float((predict != by.cpu().data.numpy()).astype(int).sum()) / float(by.size(0)))
      train_loss += loss.item()
      train_correct += float((predict == by.cpu().data.numpy()).astype(int).sum()) / float(by.size(0))
    history['train_acc'].append(train_correct/len(train_loader))
    history['train_loss'].append(train_loss/len(train_loader))

    #record accurracy
    model.eval() #tun model into evaluation mode(disable dropout, gradient)
    test_loss=0
    test_correct=0
    for datas,labels in test_loader:
      datas=Variable(datas).to(device)
      labels=Variable(labels)
      outputs = model(datas)[0]
      loss = loss_func(outputs, labels.to(device))
      predicted = torch.max(outputs, 1)[1].cpu().data.numpy()
      #test_loss +=  (float((predicted != labels.cpu().data.numpy()).astype(int).sum()) / float(labels.size(0)))
      test_loss += loss.item()
      test_correct += float((predicted == labels.cpu().data.numpy()).astype(int).sum()) / float(labels.size(0))
      if (epoch == (EPOCH-1)): #the last epoch
        tmp=[(predicted[i],labels.data.numpy()[i]) for i in range(0,len(predicted),1)]
        for item in tmp:
          history['matrix'][item[0]][item[1]]+=1
    history['valid_loss'].append(test_loss/(len(test_loader)))
    history['valid_acc'].append(test_correct/len(test_loader))

    print('Epoch:{} | train accuracy:{} | train loss:{} | test accuracy:{} | test loss:{}'.format(EPOCH,history["train_acc"][-1],history["train_loss"][-1],history["valid_acc"][-1],history["valid_loss"][-1]))
  if(device=='cuda'):
    torch.cuda.empty_cache() #clear gpu ram usage to prevent memory leak
  return history


def draw_curve(history):
  """
  draw loss urve and learning curve
  """
  EPOCH=len(history['train_acc'])
  plt.figure()
  plt.subplot(1,2,1)
  plt.plot(range(EPOCH),history['train_acc'],'x-')
  plt.plot(range(EPOCH),history['valid_acc'],'o-')
  plt.legend(['training_accuracy','testing_accuracy'])
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.title('learning_curve')
  plt.subplot(1,2,2)
  plt.plot(range(EPOCH),history['train_loss'],'-x')
  plt.plot(range(EPOCH),history['valid_loss'],'-o')
  plt.legend(['training_loss','testing_loss'])
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('loss_curve')
  plt.show()

def draw_matrix(mat):
  """
  draw confusion matrix
  """
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

  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #evaluation whether the device can use cuda
  if(device=='cuda'):
    torch.cuda.empty_cache()#to prevent gpu memory leak

  #hyperparameters
  root='./minst/'
  DOWNLOAD_MNIST = True  #true or false to download mnist data
  num_worker =0
  EPOCH = 30
  BATCH_SIZE = 2500
  LR = 0.001
  draw_graph=True
  model_path='./cnn.pth'

  print(device)

  train_data,test_data = MNIST_datasets(root,DOWNLOAD_MNIST) #create dataset
  train_loader = loader(train_data,BATCH_SIZE,num_worker,device,True) #make train loader
  valid_loader = loader(test_data,BATCH_SIZE,num_worker,device,True) #make test loader

  #main
  if mode=='train':
    cnn=CNN() #build an instance of CNN class
    history=training(cnn,device,EPOCH,BATCH_SIZE,LR,train_loader,valid_loader)

    #dump t_accuracy_l,t_loss_l,v_ac_l,v_loss_l,matrix to file for further use(draw graph in the future)
    if not ('result' in os.listdir()): #check is the folder exist
      os.mkdir('./result')
    pickle.dump(history,open("./result/history.bin",'wb'))
    if(in_colab):
      os.system('rm ./result/epoch*.txt') #remove ./result/epoch*.txt file
    else:
      os.system('del ./result/epoch*.txt') #remove ./result/epoch*.txt file

    with open('./result/epoch'+str(EPOCH)+".txt",'w') as f:
      """
      write epoch,batch,learning rate to the file
      for furture recognize
      """
      f.write('epoch:'+str(EPOCH)+"\n")
      f.write('batch:'+str(BATCH_SIZE)+"\n")
      f.write('Learning Rate:'+str(LR)+"\n")
    cnn.to('cpu')  #turn model to cpu device
    torch.save(cnn, './result/'+model_path) #save full model include structure
    zpath='./training_'+str(datetime.now()).replace("-","").replace(" ","_").replace(":","_").split(".")[0]+'.zip'
    if(in_colab):
      os.system('rm ./colab_training.zip') #remove zip file
      os.system('zip -r '+zpath+' ./result') #zip ./result folder into zip file
      files.download(zpath) #download file
    else:
      os.system('del ./colab_training.zip')
      os.system('powershell Compress-Archive ./result '+zpath)
      if('record' not in os.listdir()):
        os.mkdir('record')
      os.system('copy '+zpath+ './record/'+zpath)
  elif mode=='load':
    #check file exist-->(upload?)-->load
    if 'result' not in os.listdir(): #check if the folder exist
      if(in_colab):
        print('please upload colab_training.zip')
        zips = files.upload() #upload the zip file
        a=[i for i in zips.keys()][0] #flatten filenames keys to 1d array
        if a.endswith('.zip'): #check is it a zip file

          #a temp file to communicate python and the terminal
          with open('tmp.txt','w') as t:
            t.write(a) #(python) write filename message
          !a=$(cat tmp.txt)
          #(terminal) read filename message
          !unzip "$a"
          #(terminal) unzip
          os.remove('tmp.txt') #remove temp file
      else:
        print('please copy a colab_training.zip file to the work folder')
        input()
        exit()
    cnn     =torch.load(open('./result/'+model_path,'rb'),map_location=torch.device('cpu')) #load model, map in to cpu structure
    cnn.eval() #turn to evaluation mode

    #load train/validation's loss/accurracy data and confussion matrix
    history = pickle.load(open("./result/history.bin",'rb'))
    t_accuracy_l = history['train_acc']
    t_loss_l   = history['train_loss']
    v_ac_l    = history['valid_acc']
    v_loss_l   = history['valid_loss']
    matrix    = history['matrix']

  if draw_graph: #rather to draw graph
    draw_curve(history)
    draw_matrix(history['matrix'])


class GradCAM:
  def __init__(self,model,device): #constructor
    self.model=model
    self.model.to(device).eval()
    self.feature_maps=[]
    self.gradients=[]
    self.device=device #以備不時之需
  def get(self):
    return self.feature_maps,self.gradients

  def forward_hook(self, module, input, output): #forward hook to fetch feature map, do not modify
    self.feature_maps.append(output)

  def backward_hook(self, module, grad_input, grad_output):#backward hook to fetch gradiet data, do not modify
    self.gradients.append(grad_output[0])

  def register_hooks(self): #register hook to model
    self.feature_maps = []
    self.gradients = []
    for module in self.model.children():  #每一層
      module.register_forward_hook(self.forward_hook)
      module.register_backward_hook(self.backward_hook)
    #self.model.conv2.register_forward_hook(self.forward_hook)
    #self.model.conv2.register_backward_hook(self.backward_hook)

  def generate_cam(self,input_img,target_class):
    #preset
    self.model.zero_grad()
    self.model.eval()

    #for hook to record
    ##get result
    pred = self.model(input_img)[0]  #probibility array
    target = torch.tensor([target_class]).to(self.device)
    ##loss
    loss = F.cross_entropy(pred,target)
    ##loss backward
    loss.backward()

    gradients = self.gradients[0].to(self.device)
    # 確保 gradients 張量至少有3個維度
    if gradients.dim() < 3:
      gradients = gradients.unsqueeze(2).unsqueeze(3)  # 添加兩個維度 #batch and ?
    pooled_gradients = F.adaptive_avg_pool2d(gradients, (1, 1)) #average pooling
    # 移除多餘的維度
    #pooled_gradients = pooled_gradients[0, :, :, :]
    pooled_gradients = pooled_gradients.squeeze(0)

#################adjust index########################
######################################################
    feature_maps = self.feature_maps[2].to(input_img.device)
    for i in range(gradients.size(1)):
      feature_maps[:, i, :, :] *= pooled_gradients[i, :, :]

    heatmap = torch.mean(feature_maps, dim=1, keepdim=True)
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    #插值法填補cam的空缺
    cam = F.interpolate(heatmap, size=(input_img.size()[-2], input_img.size()[-1]), mode="bilinear", align_corners=False)
    cam = cam[0, 0, :, :]

    return cam.detach().cpu().numpy()


def load_image(dataset,index,device): #load test image
  image,label = dataset[index]
  image=image.to(device).unsqueeze(0)
  return image, label

def to_opencv(test_image,heatmap): #turn image to opencv
  heatmap = cv2.resize(heatmap, (test_image.size()[-1], test_image.size()[-2]))
  heatmap = np.uint8( 255*heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  return heatmap

def combine_graph(test_image, heatmap): #combine original img and heatmap together
  original_image = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
  superimposed_image = heatmap * 0.3 + original_image * 0.7
  superimposed_image /= superimposed_image.max()
  return superimposed_image

def show_graph(original_image,test_label,superimposed_image,predicted_class): #show graph
  plt.figure(figsize=(10, 5))
  plt.subplot(121)
  plt.imshow(original_image, cmap='gray')
  plt.title(f"True Class: {test_label}")
  plt.axis('off')
  plt.subplot(122)
  plt.imshow(superimposed_image)
  plt.title(f"Predicted Class: {predicted_class}")
  plt.axis('off')
  plt.show()

def use_gradcam(model,index,test_data,device):

  # 隨機選  取一張測試圖片，顯示其原圖和 Grad-CAM 結果
  gradcam = GradCAM(model,device) #create instance
  gradcam.register_hooks() #register hooks

  if index==None: #random
    index = np.random.choice(len(test_data))
  else:
    index=index
  test_image, test_label = load_image(test_data,index,device)

  # 確保模型處於評估模式，以避免 dropout 等層的影響
  cnn.eval()

  # 取得模型預測的類別
  with torch.no_grad():
    predicted_class = cnn(test_image)[0].argmax().item()

  # 生成 Grad-CAM
  heatmap = gradcam.generate_cam(test_image, target_class=predicted_class)

  # 將 heatmap 轉為 OpenCV 格式
  heatmap = to_opencv(test_image,heatmap)

  # 將 heatmap 與原圖結合
  superimposed_image = combine_graph(test_image,heatmap)

  # 顯示圖像和 Grad-CAM 結果
  org_img=test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
  show_graph(org_img,test_label,superimposed_image,predicted_class)

  featuremap,gradients=gradcam.get() #for debug

index=None
use_gradcam(cnn,index,test_data,device)
use_gradcam(cnn,index,test_data,device)
use_gradcam(cnn,index,test_data,device)
