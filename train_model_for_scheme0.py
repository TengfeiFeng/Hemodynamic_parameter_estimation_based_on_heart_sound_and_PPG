# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 2020

@author: Feng,Tengfei, Dalian University of Technology, School of Biomedical Engineering
"""


import os

import scipy.io as sio
from keras import backend as K

import numpy as np

from keras.layers import GRU, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation,  Input, CuDNNGRU
from keras.utils import multi_gpu_model
from keras.models import Model
from keras import initializers, regularizers
from keras.layers import Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from tqdm import tqdm
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

   
def load_data(input_directory):
# recoed location of data
    data = []

    for folder in os.listdir(input_directory):
        tmp_folder = os.path.join(input_directory,folder)
        for f in os.listdir(tmp_folder):
            if os.path.isfile(os.path.join(tmp_folder, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                data.append(os.path.join(tmp_folder, f))       
    return data
def load_data_batch(data_path,maxlen,batch_size):
# load batch data for train or test
    data = []
    label = []

    for i in range(len(data_path)):
        data.append(np.transpose(sio.loadmat(data_path[i])['pcg_ppg']))
        lab = sio.loadmat(data_path[i])['label'][0]
        lab[2:] = np.round(lab[2:],1)
        lab[0:2] = np.round(lab[0:2],2)
        lab[0] = lab[0]/50
        lab[1] = lab[1]/10
        lab[2] = lab[2]/2000
        lab[3] = lab[3]/1000
        label.append(lab)
    data = np.array(data).reshape(batch_size,maxlen,2)
    return data,np.array(label)

def _bn_relu(layer, dropout=0, **params):
    layer = Activation('relu')(layer)

    if dropout > 0:
        layer = Dropout(0.5)(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D 
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same')(layer)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    # the code of building resnet block was reference to availabel code used in A. Y. Hannun et al., 
    #Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms 
    #using a deep neural network, Nat. Med., vol. 25, no. 1, pp. 65-69, Jan. 2019.
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length,padding="same")(layer)
    zero_pad = (block_index % 4) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(2):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=0.5 if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            16,
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
        layer= BatchNormalization()(layer)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / 4) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        16,
        32,
        subsample_length=1,
        **params)
    layer = BatchNormalization()(layer)
    layer = _bn_relu(layer, **params)
    conv_subsample_lengths = [1,2,1,2,1,2]
    for index, subsample_length in enumerate(conv_subsample_lengths):
        num_filters = get_num_filters_at_index(
            index, 32, **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def load_data_batch_without_rescale(data_path,maxlen,batch_size):
    # load data without rescale
    data = []
    label = []
    for i in range(len(data_path)):
        data.append(np.transpose(sio.loadmat(data_path[i])['pcg_ppg']))
        lab = sio.loadmat(data_path[i])['label'][0]
        lab[2:] = np.round(lab[2:],1)
        lab[0:2] = np.round(lab[0:2],2)
        label.append(lab)
    data = np.array(data).reshape(batch_size,maxlen,2)
    return data,np.array(label)

def result_fix(data):
    # fix the estimated result 
    data[:,0] = data[:,0]*50
    data[:,1] = data[:,1]*10
    data[:,2] = data[:,2]*2000
    data[:,3] = data[:,3]*1000
    return data

if __name__ == '__main__':
    # split data for 5-fold validation 
    TRAIN_DATA_PATH1 = 'data/subject_1'
    TRAIN_DATA_PATH2 = 'data/subject_2'
    data1 = load_data(TRAIN_DATA_PATH1)
    data2 = load_data(TRAIN_DATA_PATH2)
    train_data = np.hstack([data1,data2])
    np.random.shuffle(train_data)
    train_data2,train_data1 = train_test_split(train_data,test_size=0.2)
    train_data3,train_data2 = train_test_split(train_data2,test_size=0.25)
    train_data4,train_data3 = train_test_split(train_data3,test_size=0.3333) 
    train_data5,train_data4 = train_test_split(train_data4,test_size=0.5)
    np.save('data1.npy',train_data1)
    np.save('data2.npy',train_data2)
    np.save('data3.npy',train_data3)
    np.save('data4.npy',train_data4)
    np.save('data5.npy',train_data5)
    del data1,data2,train_data,train_data1,train_data2,train_data3,train_data4,train_data5
    # to record epoch number of the best validation model
    for i in range(1,6):
        train_data= np.array([])
        val_data = []
        # select the ith sub dataset as hidden test data
        test_data = np.load('data'+str(i)+'.npy')
        # the remained four data as the train data
        for j in range(1,i):
            train_data = np.hstack([train_data,np.load('data'+str(j)+'.npy')])
        if i<5:
            for j in range(i+1,6):
              train_data = np.hstack([train_data,np.load('data'+str(j)+'.npy')]) 
        # randomly select 25% of train data as the validation set
        train_data,val_data = train_test_split(train_data,test_size=0.25)
        # location for save trained model
        filepath = 'saved_model_for_sheme0/'+str(i)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        # number of estimated parameters and the input length
        num_of_parameters = 4
        max_len = 1000
        # build the model
        main_input = Input(shape=(1000,2), dtype='float32', name='main_input')
        # add residul network
        x = add_resnet_layers(main_input)
        # add Bi-GRU modeule
        x = Bidirectional(CuDNNGRU(32,return_sequences=False,return_state=False))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        # output
        main_output = Dense(num_of_parameters)(x)
        model=Model(inputs=main_input, outputs=main_output)
        
        # loss function and optimazation method
        optimizer = Adam(
            lr=0.001,
            clipnorm=1)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])
    #        
        model.summary()
        train_loss = []
        train_mae = []
        val_loss = []
        val_mae = [] 
        # set batch_size
        batch_size = 64
        for epoch in tqdm(range(int(np.ceil(np.size(train_data)/batch_size))*200)):
            epoch1 = epoch%int(np.size(train_data)/batch_size)
            if  epoch1 == 0:
                # after training an epoch, shuffle the train data
                np.random.shuffle(train_data)
            if epoch1 == 0: 
                mae = []
                loss = []
                # observe the performace on validation set for every epoch
                val,v_label = load_data_batch(val_data,max_len,len(val_data))
                gg = model.evaluate(val,v_label)
                mae.append(gg[1])
                loss.append(gg[0])
                del val,v_label,gg
                print("Test acc:",np.mean(mae))
                print("Test loss:",np.mean(loss))#/len(val))
                val_loss.append(np.mean(loss))
                val_mae.append(np.mean(mae))
                # save the best validation model with smallest 
                if val_loss[-1]==min(val_loss):
                    model_name = 'smallest_mse_model'
                    model.save_weights(os.path.join(filepath,model_name))
                if val_mae[-1]==min(val_mae):
                    model_name = 'smallest_mae_model'  
                    model.save_weights(os.path.join(filepath,model_name))
            data,label = load_data_batch(train_data[epoch1*batch_size:(epoch1+1)*batch_size],max_len,batch_size)
            his = model.train_on_batch(data,label)
            train_loss.append(his[0])
            train_mae.append(his[1])
            print(his)
    
 # then load the best validation models with smallest mae values to evaluate on the corresponding hidden test data, respectively

    for i in range(1,6):
        filepath = 'saved_model_for_sheme0/'+str(i)
        model.load_weights(os.path.join(filepath,'smallest_mae_model'))
        test_data_path = 'data'+str(i)+'.npy'
        test_data = np.load(test_data_path)
        test,test_label= load_data_batch_without_rescale(test_data,1000,len(test_data))
        test_pred = result_fix(model.predict(test))
        test_mae_error = np.mean(abs(test_label-test_pred),0)
        test_std = np.std(test_label-test_pred,0)
        test_me = np.mean(test_pred-test_label,0)
        print('fold -',str(i),'results:')
        print('MAE error:')
        print(np.round(test_mae_error,3))
        print('SD:')
        print(np.round(test_std,3))
        print('ME error:')
        print(np.round(test_me,3))
        
