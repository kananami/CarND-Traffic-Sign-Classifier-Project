"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import random
import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import data_handler as dh
from vgg_bn import VGGnet
import numpy as np
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.5            # learning rate
DOWNLOAD_MNIST = False

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Data.Dataset):
    def __init__(self,root='./data',train='train',transform=None,target_transform=None,loader=default_loader):
        self.imgs = []
        if train == 'train':
            X_train,y_train = dh.get_train_data(root)
            #print(np.append(X_train.reshape(-1,32*32*3),y_train.reshape(-1,1)))
            self.data = X_train.reshape(-1,32*32*3)
            self.labels = y_train.reshape(-1,1)
            # #print(imgs.shape)
            # for i in range(self.y_train.shape[0]):                       #I've tried to get rid of this loop,but failed
            #     self.imgs.append((self.X_train[i],self.y_train[i]))
            # self.data,self.labels = self.imgs[0],self.imgs[1]
        elif train =='test':
            X_test,y_test = dh.get_test_data(root)
            self.data = X_test.reshape(-1,32*32*3)
            self.labels = y_test.reshape(-1,1)
            # for i in range(self.y_test.shape[0]):
            #     self.imgs.append((self.X_test[i],self.y_test[i]))
            # self.data,self.labels = self.imgs[0],self.imgs[1]
        elif train =='valid':
            X_valid,y_valid = dh.get_valid_data(root)
            self.data = X_valid.reshape(-1,32*32*3)
            self.labels = y_valid.reshape(-1,1)
            # for i in range(self.y_valid.shape[0]):
            #     self.imgs.append((self.X_valid[i],self.y_valid[i]))
            # self.data,self.labels = self.imgs[0],self.imgs[1]
        self.n_data = self.data.shape[0]
        self.n_label = self.labels.shape[0]
        self.n_classes = max(self.labels)+1
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, item):
        # get train data and label frome index in the dataset
        #img, label = self.imgs[item]
        img,label = self.data[item],self.labels[item]
        img = Image.fromarray(img.reshape(32,32,3))
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return self.data.shape[0]

transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

train_data = MyDataset(root=r'./data',train='train',transform=transform)
# plot one example
print(train_data.n_data)                 # (60000, 28, 28)
print(train_data.n_label)               # (60000)
print(train_data.n_classes)           # (43)
with open('signnames.csv','r') as f:
    signnames = f.read()
# for line in signnames[1:]:
#     print(line.split(',')[0])
#     if line ==1:
#         break
id_to_name = {int(line.split(',')[0]):line.split(',')[1] for line in signnames.split('\n')[1:] if len(line)>0}
random_index = [random.randint(0,train_data.n_data) for _ in range(3*3)]
fig = plt.figure(figsize=(15,15))
for i,index in enumerate(random_index):
    a = fig.add_subplot(3,3,i+1)
    implot = plt.imshow(train_data.data[index].reshape(32,32,3))
    a.set_title('%i,%s' %(train_data.labels[index], id_to_name[int(train_data.labels[index])]))

# plt.imshow(train_data.data[40].reshape(32,32,3), cmap='gray')
# plt.title('%i,%s' %(train_data.labels[0], id_to_name[int(train_data.labels[0])]))
plt.show()

fig,ax = plt.subplots()
values,bins,patches = ax.hist(train_data.labels,train_data.n_classes,normed=10)
ax.set_xlabel('Smarts')
ax.set_title(r'Histogram of classess')
fig.tight_layout()
print('Most common index')
most_common_index = sorted(range(len(values)),key=lambda k:values[k],reverse=True)
for index in most_common_index[:10]:
    print('index:%s=> %s=%s' %(index,id_to_name[index],values[index]))
plt.show()
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

test_dataset = MyDataset(root=r'./data',train='test',transform=transform)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=50,
)

# convert test data into Variable, pick 2000 samples to speed up testing

#
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 64, 64)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
                                            # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 16, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 32, 32)
        )
        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
                                            # output shape (32, 7, 7)
        )
        self.conv4 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 16, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 16, 16)
        )
        self.conv5 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16,32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
                                           # output shape (32, 7, 7)
        )
        self.conv6 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 8, 8)
        )
        self.conv7 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 2),     # output shape (, 14, 14)
            nn.ReLU(),                      # activation
                                            # output shape (32, 7, 7)
        )
        self.conv8 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(64, 64, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
                                            # output shape (32, 7, 7)
        )
        self.conv9 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(64, 64, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 4, 4)
        )
        self.fc= nn.Sequential(
            nn.Linear(64*4*4,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.out = nn.Linear(1024, 43)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        out= self.fc(x)   # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(out)
        return output, x    # return x for visualization

####-------------------------------------------------

#my_vgg = VGGnet(num_classses=43)
if __name__=='__main__':
    my_vgg =CNN()
    print(my_vgg)
    optimizer = torch.optim.Adam(my_vgg.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # following function (plot_with_labels) is for visualization
    from matplotlib import cm
    try:from sklearn.manifold import TSNE;HAS_SK = True
    except:HAS_SK=False;print('Please install sklearn for layer visualization')
    def plot_with_labels(lowDWeights,labels):
        plt.cla()
        X,Y = lowDWeights[:,0],lowDWeights[:,1]
        for x,y,s in zip(X,Y,labels):
            c = cm.rainbow(int(255*s/9));plt.text(x,y,s,backgroundcolor=c,fontsize=9)
        plt.xlim(X.min(),X.max());plt.ylim(Y.min(),Y.max());plt.title('Visualize last layer');plt.show();plt.pause(100)
    plt.ion()
    for epoch in range(EPOCH):
        #print(epoch)
        for step,(x,y) in enumerate(train_loader):
            if step >0:
                break
            b_x = Variable(x)
            b_y = Variable(torch.squeeze(y.long()))
            output = my_vgg(b_x)[0]

            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50 ==0:
                for step,(x_test,y_test) in enumerate(test_loader):
                    if step>0:
                        break
                    x_test = Variable(x_test)
                    y_test = torch.squeeze(y_test.long())   #type(y_test) is <class 'torch.LongTensor'> torch.Size([50])
                    test_output,last_layer = my_vgg(x_test)  # test_output is a type of Variable
                    pred_v = torch.max(test_output,1)[1].data.squeeze()   #type(pred_v) is  <class 'torch.LongTensor'> torch.Size([50])
                    accuracy = sum(pred_v==y_test)/float(y_test.size(0))
                    print('Epoch:',epoch,'|train loss:%.4f'%loss.data[0],'|test accuracy: %.2f'%accuracy)
                    if HAS_SK:
                        tsne = TSNE(perplexity=30.0,n_components=2,init='pca',n_iter=5000)
                        plot_only = 50
                        print(last_layer.shape)
                        low_dim_embs =tsne.fit_transform(last_layer.data.numpy()[:plot_only,:])

                        labels = y_test.numpy()[:plot_only]

                        plot_with_labels(low_dim_embs,labels)



    plt.ioff()


############-----------------------------------------------------------------------------------