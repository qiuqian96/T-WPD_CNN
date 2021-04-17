#!/usr/bin/env python
# coding: utf-8

# Author: Qian Qiu
# Year: 2021
# Language: python 3


from keras.models import Sequential,Model
from keras import layers
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt
import pywt
import cv2

def loadData(filePath):
#     load data from filePath
    csvData = pd.read_csv(filePath)
    data = np.array(csvData)
    return data

def augment_data(data, label, num_augmentations):
#    Take data and label, and shift a random choice of the data, padding with zeroes, to produce an augmented dataset
#    This returns just the extra data generated, not the concatenation of the original and the extra.
    extra_data = []
    extra_label = []
    num_unaugmented = len(label)
    image_width = len(data[0][0][0])
    max_shift = int (image_width * 0.1 - 1.5)
    for i in range(num_augmentations):
        data_to_shift = np.random.randint(0, num_unaugmented)
        shift_by = np.random.randint(-max_shift, max_shift + 1)
        shifted = np.roll(data[data_to_shift][0], shift_by, axis = 1)
        shifted = np.array([shifted])
        extra_data.append(shifted)
        extra_label.append(label[data_to_shift])
    return (np.array(extra_data), np.array(extra_label))

def waveletPacket(data):
#     Get wavelet packet decomposition coefficient
#     Divided into 3 layers
#     'aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'
    n=3
    num = len(data)
    height = data.shape[2]
    width = data.shape[3]
    recof = []
    for i in range(num):
        for j in range(height):
            wp = pywt.WaveletPacket(data[i][0][j], wavelet='db8',maxlevel = n)
            for k in [node.path for node in wp.get_level(n, 'natural')]:
                recof.append(wp[k].data)
    recof = np.array(recof).reshape(num,1,height,-1)
    return recof

def get_TWavPadata(data):
#     Combine wavelet packet decomposition coefficients with original data into an array
    recof = waveletPacket(data)
    num = recof.shape[0]
    height = recof.shape[2]
    width_recof = recof.shape[3]
    width_data = data.shape[3]
    
#   The length of coefficients and original data are different 
    if(width_data>width_recof):
        width = width_data
    else:
        width = width_recof
    
    WavPadata = np.zeros((num,2,height,width))
    WavPadata[:,:1,:,:width_data] = data
    WavPadata[:,1:,:,:width_recof] = recof
    return WavPadata,width_data,width_recof

def Dct_Fredata(data):
#     DCT transform for frequency-domain data
#     The length in frequency domain after DCT transform is the same as that in time domain
    Fdata = []
    num = len(data)
    height = data.shape[2]
    width = data.shape[3]
    for i in range(num):
        for j in range(height):
            Fdata.append(abs(cv2.dct(data[i][0][j])))
    Fdata = (np.array(Fdata)).reshape(num,1,height,width)
    return Fdata

def get_TFdata(data):
#     Combine time and frequency domain data into an array
    Fdata = Dct_Fredata(data)
    XTFdata = np.zeros((len(data),2,data.shape[2],data.shape[3]))
    XTFdata[:,:1,:,:] = data
    XTFdata[:,1:,:,:] = Fdata
    return XTFdata

def T_WPD_cnn(inputshape1,inputshape2):
    input1 = layers.Input(inputshape1)
    input2 = layers.Input(inputshape2)
    
    cnn1 = single_cnn(inputshape1)(input1)
    cnn2 = single_cnn(inputshape2)(input2)

    combinedInput = layers.concatenate([cnn1, cnn2],axis = -1)
    
    x = layers.Flatten()(combinedInput)
    x = layers.Dense(16, activation = 'relu')(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)
    adam = Adam(lr = 0.001)
    model = Model(inputs = [input1, input2],outputs = x)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

    return model

def single_cnn(inputshape):
#     single channel of T_WPD_cnn
    height = inputshape[1]
    
    inputs = layers.Input(inputshape)
    x = layers.Conv2D(16,(height,10),data_format = 'channels_first',activation = 'relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.pooling.MaxPooling2D((1,2),data_format = 'channels_first',strides=(1,2))(x)
    
    x = layers.Conv2D(8,(1,10),data_format = 'channels_first',activation = 'relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.pooling.MaxPooling2D((1,2),data_format = 'channels_first',strides=(1,2))(x)
    
    model = Model(inputs,outputs)
    return model


filePath = './nor_mine_data.csv'
seismic_data = loadData(filePath)
num_inputs = seismic_data.shape[1]-1
data = seismic_data[:, 0:num_inputs]
label = seismic_data[:, num_inputs]

image_num = data.shape[0]
image_channels = 1
image_height = 1
image_width = 1000
data = data.reshape(image_num, image_channels, image_height, image_width)

#parameters prepare
seed = 7
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
fold_num = 0

total_TT = 0
total_TF = 0
total_FT = 0
total_FF = 0
AUC = 0

# choose data type
data_is_T_WPD = False
data_is_TF = True

for train, test in kfold.split(data, label):
    fold_num+=1
   
    #    split off 10% of the training data to act as validation
    data_train, data_validation, label_train, label_validation = train_test_split(data[train], label[train], test_size = 0.1,
                                                                                  random_state = seed, stratify = label[train])

#    obtain the index array that the label equal to 0 or 1
    indices_where_label_train_zeros = np.where(label_train == 0)[0]
    indices_where_label_train_one = np.where(label_train == 1)[0]
    
#    generate the same number of ones and zeros in the training set
    num_excess = len(indices_where_label_train_zeros) - len(indices_where_label_train_one)
    data_gen, label_gen = augment_data(data_train[indices_where_label_train_one],label_train[indices_where_label_train_one],num_excess)
    data_train = np.concatenate((data_train, data_gen))
    label_train = np.concatenate((label_train, label_gen))
    
#   trim the training set by removing from each end
    num_to_remove = int(len(data_train[0][0][0]) * 0.1 - 0.5)
    data_train = np.delete(data_train, np.s_[:num_to_remove], axis = 3)
    data_train = np.delete(data_train, np.s_[-num_to_remove:], axis = 3)

    data_validation = np.delete(data_validation, np.s_[:num_to_remove], axis = 3)
    data_validation = np.delete(data_validation, np.s_[-num_to_remove:], axis = 3)
    
    data_test = data[test]
    label_test = label[test]
    data_test = np.delete(data_test, np.s_[:num_to_remove], axis = 3)
    data_test = np.delete(data_test, np.s_[-num_to_remove:], axis = 3)
    
    if(data_is_T_WPD):
    #     Convert to time and WPD coefficient data
        data_train,width_data,width_recof = get_TWavPadata(data_train)
        data_validation,width_data,width_recof = get_TWavPadata(data_validation)
        data_test,width_data,width_recof = get_TWavPadata(data_test)
    #     train cnn model
        cnn_model = T_WPD_cnn(inputshape1=(1,data_train.shape[2],width_data),inputshape2=(1,data_train.shape[2],width_recof))
    #     early stop trained model
        es = EarlyStopping(monitor = 'val_loss', min_delta = 0.0, patience = 10, verbose = 0, mode = 'min')
        history = cnn_model.fit([data_train[:,:1,:,:width_data],data_train[:,1:,:,:width_recof]], label_train,
                        validation_data = ([data_validation[:,:1,:,:width_data],data_validation[:,1:,:,:width_recof]], label_validation),
                        epochs = 50, batch_size = 50, verbose = 0, callbacks = [es])
        
    if(data_is_TF):
    #     Convert to time and frequency domain data
        data_train = get_TFdata(data_train)
        data_validation = get_TFdata(data_validation)
        data_test = get_TFdata(data_test)
    #     train cnn model
        cnn_model = T_WPD_cnn(inputshape1=(1,data_train.shape[2],data_train.shape[3]),
                              inputshape2=(1,data_train.shape[2],data_train.shape[3]))
    #     early stop trained model
        es = EarlyStopping(monitor = 'val_loss', min_delta = 0.0, patience = 10, verbose = 0, mode = 'min')
        history = cnn_model.fit([data_train[:,:1],data_train[:,1:]], label_train,
                            validation_data = ([data_validation[:,:1],data_validation[:,1:]], label_validation),
                            epochs = 50, batch_size = 50, verbose = 0, callbacks = [es])
   
    if fold_num == 1:
        cnn_model.summary()
        print("\nFold: 10" + "\tEpochs: 50" + "\tUsing EarlyStopping\n")
    print ('Fold>>',fold_num)
    for ep in history.epoch:
        if ((ep+1)%5) == 0:
            print ('>> Epoch:', ep+1,'/',len(history.epoch),' >>> Loss:',history.history['loss'][ep], ' >>> Acc:', history.history['accuracy'][ep])
    
#    cnn model test    
    if(data_is_T_WPD):
        label_predict = np.round(cnn_model.predict([data_test[:,:1,:,:width_data],data_test[:,1:,:,:width_recof]]))
    if(data_is_TF):
        label_predict = np.round(cnn_model.predict([data_test[:,:1],data_test[:,1:]]))

#    calculate the accuracy
    num_TT = 0
    num_TF = 0
    num_FT = 0
    num_FF = 0

    for i in range(len(label_test)):
        if label_test[i] == 1:
            if label_predict[i] == 1:
                num_TT += 1
            else:
                num_TF += 1
        else:
            if label_predict[i] == 0:
                num_FF += 1
            else:
                num_FT += 1
                
    print("Number of predicted events: " + str(len(label_test)))
    print("Number of test true events: " + str(num_TT + num_TF))
    print("Number of predicted true events: " + str(num_TT + num_FT))
    
    if (num_TT + num_TF) > 0:
        tpRate = num_TT / (num_TT + num_TF)
        print("Accuracy rate of TT: %0.4f" % tpRate)
        print("Accuracy rate of TF: %0.4f" % (num_TF / (num_TT + num_TF)))
            
    if (num_FT + num_FF) > 0:
        print("Accuracy rate of FT: %0.4f" % (num_FT / (num_FT + num_FF)))
        print("Accuracy rate of FF: %0.4f" % (num_FF / (num_FT + num_FF)))
        
    total_TT += num_TT
    total_TF += num_TF
    total_FT += num_FT
    total_FF += num_FF
    
#     AUC
    if(data_is_T_WPD):
        scores = cnn_model.predict([data_test[:,:1,:,:width_data],data_test[:,1:,:,:width_recof]])
    if(data_is_TF):
        scores = cnn_model.predict([data_test[:,:1],data_test[:,1:]])
    fpr, tpr, thresholds = metrics.roc_curve(label_test, scores, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    AUC = AUC + auc
    print("The %dth fold AUC: %0.3f" % (fold_num,auc))

print("\nSUMMARY")
print("Total number of test true events = " + str(total_TT + total_TF))
print("Total number of predicted true events = " + str(total_TT + total_FT))
print("Average accuracy rate of TT: %0.4f" % (total_TT / (total_TT + total_TF)))
print("Average accuracy rate of TF: %0.4f" % (total_TF / (total_TT + total_TF)))
print("Average accuracy rate of FT: %0.4f" % (total_FT / (total_FT + total_FF)))
print("Average accuracy rate of FF: %0.4f" % (total_FF / (total_FT + total_FF)))
print("The average AUC: %0.3f" % (AUC/10))

