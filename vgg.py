from torchvision import transforms,utils
import data_handler as dh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from PIL import Image
import math
from torch.autograd import Variable
EPOCHS = 1
LR = 0.01
BATCH_SIZE = 20   #there is 256 in VGG Nural Network
# VGG网络定义

#
class VGGnet(nn.Module):
    def __init__(self,num_classses=1000,init_weights=True):
        super(VGGnet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #（32,14,14)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU()
        )
        self.conv4= nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU()
        )
        self.conv8=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classses),

        )
        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv12(x)
        x = self.conv15(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x =x.view(x.size(0),-1)
        output = self.classifier(x)
        return output,x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m,nn.Linear):
                    m.weight.data.normal_(0,0.01)
                    m.bias.data.zero_()

# data loader

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Data.Dataset):
    def __init__(self,root='./data',train='train',transform=None,target_transform=None,loader=default_loader):
        fh = open(root,'r')
        imgs =[]
        for line in fh:
            line =line.strip('\n')
            line = line.rstrip()
            words= line.split('\t')
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img =self.loader(fn)
        # img = Image.fromarray(self.imgs[item])
        # label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)
#  I have resize train data in helper function.   # but it isn't the best way to do
traindata = r'./data/train.txt'
#root =r'./data'
train_dataset = MyDataset(traindata,transform=transform)
train_loader =Data.DataLoader(
    dataset =train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
testdata = './data/test.txt'
test_dataset = MyDataset(testdata,transform)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=10,
)

def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transposs((1,2,0)))
    plt.title("Batch from dataloader")
if __name__=='__main__':
    #test_model
    my_vgg = VGGnet(num_classses=43)
    print(my_vgg)
    optimizer = torch.optim.Adam(my_vgg.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        #print(epoch)
        for step,(x,y) in enumerate(train_loader):
            #print(x.shape,y.shape)
            b_x = Variable(x)
            b_y = Variable(y)
            #print(b_x.shape,b_y.shape)
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

                    test_output,last_layer = my_vgg(x_test)
                    pred_v = torch.max(test_output,1)[1].data.squeeze()
                    accuracy = sum(pred_v==y_test)/float(y_test.size(0))
                    print('Epoch:',epoch,'|train loss:%.4f'%loss.data[0],'|test accuracy: %.2f'%accuracy)


