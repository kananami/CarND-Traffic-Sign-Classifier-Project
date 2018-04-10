#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
from skimage import io
import cv2
from PIL import Image


TRAIN_FILE = "train.p"
VALID_FILE = "valid.p"
TEST_FILE = "test.p"

def get_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    # Load the dataset
    training_file = os.path.join(folder, TRAIN_FILE)
    validation_file= os.path.join(folder, VALID_FILE)
    testing_file =  os.path.join(folder, TEST_FILE)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Retrive all datas
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_torch_vgg_data():
    root =r'./data'
    imges = {}

    from pandas.io.parsers import read_csv
    X_train, y_train, X_valid, y_valid, X_test, y_test=get_data(root)
    classes = read_csv('signnames.csv').values[:,1]
    #print(classes)
    #print(y_train)
    torch_traindata_path = r'./data/train.txt'
    with open(torch_traindata_path,'w') as f:
        for i,img in enumerate(X_train):
            img = cv2.resize(img,(224,224))
            img_path=os.getcwd()+'/data/torchdata/train/'+str(i)+'.jpg'
            io.imsave(img_path,img)
            f.write(img_path+'\t'+str(y_train[i])+'\n')
            if i% 500 ==0:
                print(str(i/len(X_train)*100)+'%')
    print('train data done!')
    with open(r'./data/test.txt','w') as f:
        for i,img in enumerate(X_test):
            img = cv2.resize(img,(224,224))
            img_path=os.getcwd()+'/data/torchdata/test/'+str(i)+'.jpg'
            io.imsave(img_path,img)
            f.write(img_path+'\t'+str(y_test[i])+'\n')
            if i% 500 ==0:
                print(str(i/len(X_test)*100)+"%")
    print('train data done!')
    with open(r'./data/valid.txt','w') as f:
        for i,img in enumerate(X_valid):
            img = cv2.resize(img,(224,224))
            img_path=os.getcwd()+'/data/torchdata/valid/'+str(i)+'.jpg'
            io.imsave(img_path,img)
            f.write(img_path+'\t'+str(y_valid[i])+'\n')
            if i% 500 ==0:
                print(str(i/len(X_valid)*100)+'%')
    print('valid data done!')

#getdata()

def get_train_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    # Load the train dataset
    training_file = os.path.join(folder, TRAIN_FILE)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)


    X_train, y_train = train['features'], train['labels']


    return X_train, y_train

def get_test_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    #load the test data
    testing_file =  os.path.join(folder, TEST_FILE)


    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)


    X_test, y_test = test['features'], test['labels']

    return X_test, y_test

def get_valid_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    # Load the valid dataset
    validation_file= os.path.join(folder, VALID_FILE)

    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    X_valid, y_valid = valid['features'], valid['labels']
    return X_valid, y_valid