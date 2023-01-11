import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Flatten, Activation, Convolution2D,Input,Conv2DTranspose,GlobalAveragePooling2D,UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D,Dense,Multiply,Concatenate,ReLU,BatchNormalization,Lambda,Reshape
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import Model
import os
import random
from tensorflow.python.ops import array_ops
import sys
from Preprocessing import pre_processing,myMaskRCNNConfig,get_rcnn
from sklearn.utils import class_weight
import math
import keras.backend as K
import json
import time
from tensorflow import keras

def create_model():
    inputs=Input(shape=(224, 224, 3))
    layerc1in=BatchNormalization()(inputs)
    mulayerc1=Convolution2D(filters=1,kernel_size=(19,19),strides=(2, 2),activation='linear',padding='same')(layerc1in)
    sigmalayerc1=Convolution2D(filters=1,kernel_size=(19,19),strides=(2, 2),activation='linear',padding='same')(layerc1in)
    layerc1 = Lambda(sample_z)([mulayerc1, sigmalayerc1])
    layerc1=Reshape((112,112,1))(layerc1)
    layerc1=Convolution2D(filters=32,kernel_size=(19,19),strides=(1, 1),activation='relu',padding='same')(layerc1)
    layer1= Conv2DTranspose(16, (19,19), strides=(2, 2),padding='same',activation='relu') (layerc1)
    #layer1=UpSampling2D((2,2))(layerc1)
    outputs1=Convolution2D(filters=8,kernel_size=(19,19),strides=(1, 1),activation='relu',padding='same')(layer1)
    outputs=Convolution2D(filters=2,kernel_size=(19,19),strides=(1, 1),activation='sigmoid',padding='same')(outputs1)
    model=Model([inputs],[outputs])
    return model

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(1,tf.shape(mu)[1],tf.shape(mu)[2],tf.shape(mu)[3]), mean=0, stddev=1)
    return mu + K.exp(log_sigma / 2) * eps

kmodel = keras.models.load_model('C:\Corn_Production\Production\preweight\multiclass_weight.h5')

dic = {0 : 'CER', 
       1 : 'DER13', 
       2 : 'DER579', 
       3 : 'GER', 
       4 : 'Healthy', 
       5 : 'Other'}

def DRWAP_rate(im_path,seg_model,rate_model,print_DRWAP=False):
    ratio=np.array([[0,0],[1,0.1],[3,0.25],[5,0.5],[7,0.75],[9,1]])
    img=cv2.imread(im_path)
    print("<-------------------- Before preprocess -------------------->")
    images,prediction,ims_crop,n_cluster,sortrois=pre_processing(im_path,seg_model)
    print("<-------------------- After preprocess -------------------->")
    rates=[]
    print("Type of images ==> {d} ".format(d = type(images)))
    print("Type of prediction ==> {d}".format(d = type(prediction)))
    print("Type of ims_crop ==> {d}".format(d = type(ims_crop)))
    print("Type of n_cluster ==>{d} ".format(d = type(n_cluster)))
    print("Type of sortrois ==> {d}".format(d = type(sortrois)))
    for image in ims_crop:
        im2ar = np.asarray(image)
        im2ar = cv2.resize(im2ar, (224,224))
        im2ar = im2ar/255.
        im2ar = im2ar.reshape((1, im2ar.shape[0], im2ar.shape[1], im2ar.shape[2]))
        if "DER" not in (dic[np.argmax(kmodel.predict(im2ar),axis=1)[0]]):
            rates.append(dic[np.argmax(kmodel.predict(im2ar),axis=1)[0]])
        else:
            rate,heighest=rating_by_ear(image,rate_model,print_rate=False)
            rates.append(rate)
    rates=np.array(rates)
    (unique, counts) = np.unique(rates, return_counts=True)
    
    for isnum in range(len(unique)):
        if not (unique[isnum].isnumeric()):
            unique = np.delete(unique, isnum)
            counts = np.delete(counts, isnum)
    unique = unique.astype(int)
    ratio=np.array([x[1] for x in ratio if x[0] in unique ])
    DRWAP=np.sum(counts.T*ratio)/rates.shape[0]*100

    if print_DRWAP:
        print('rate each an ear',rates,'DRWAP of this group',str(DRWAP)+'%')
    return rates,DRWAP,sortrois

def rating_by_ear(img,model,print_rate=False):
    h_img,w_img=img.shape[0:2]
    resize_shape=model.layers[0].get_output_at(0).get_shape().as_list()[1:3]
    img=cv2.resize(img/255.,(resize_shape[0],resize_shape[1]))
    pre=model.predict(img.reshape(1,resize_shape[0],resize_shape[1],3))
    pre=np.argmax(pre[0],axis=2).astype('float32')
    pre[np.sum(img,axis=2)==0]=0
    pre=cv2.resize(pre,(w_img,h_img))
    kernel = np.ones((149,149),np.float32)/(149**2) #Kernelfilter
    dst = cv2.filter2D(pre,-1,kernel)
    dst[dst<0.5]=0
    percentder=np.mean(dst,axis=1)
    percentder[percentder<0.5]=0
    heighest=np.where(percentder>=0.5)[0]
    if heighest.shape[0]!=0:
        heighest=(np.min(heighest)+1)/percentder.shape[0]*100

        heighest=100-heighest
    else:
        heighest=0
    
    if heighest<=10:
        heighest+=1
        rate=1
    elif heighest <= 25:
        rate =3
    elif heighest <= 50:
        rate=5
    elif heighest <=75:
        rate=7
    elif heighest <= 100:
        rate=9
        
    if print_rate: #ไม่แสดงเปลี่ยน print_rate==False
        print(rate,heighest)
    
    return rate,heighest

def rbg (impath):
    #Rate by group
    model = create_model()
    model.load_weights('C:/Corn_Production/Production/preweight/rating_weight-0042.h5')
    model_dir=''
    rcnn_weight='C:/Corn_Production/Production/preweight/rcnn_weight.h5'
    config = myMaskRCNNConfig()
    #impath=r"D:\My Drive\AI\Datasets\corn_disease\corn_group\3.jpg"
    filename, file_extension = os.path.splitext(os.path.basename(impath))
    rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
   
    return rates_r,DRWAP_r,sortrois_r,impath,filename

def export_json(impath):
    rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
    # Data to be written
    js ={
        "extID": "",
        "plotID": "",
        "trialID": "",
        "barcode": "",
        "cornGroupID": "",
        "earDetail": [],
        "predicted": []}

    js["cornGroupID"]=os.path.split(impath)[1].split(".")[0]
    for i in range(sortrois_r.shape[0]):
        js["earDetail"].append({"ID":str(i+1),"position":{"xlt":str(sortrois_r[i,0]),"ylt":str(sortrois_r[i,1]),"xrb":str(sortrois_r[i,2]),"yrb":str(sortrois_r[i,3])},"rate":str(rates_r[i])})
    js["predicted"].append({"DRWAP":str(DRWAP_r)})
    dercount = 0
    for j in range(len(rates_r)):
        if rates_r[j].isnumeric():
            dercount+=1
    DLERP_r=dercount/len(rates_r)*100
    js["predicted"].append({"DLERP":str(DLERP_r)})
    js["predicted"].append({"#totalEar":str(len(rates_r))})
    js["predicted"].append({"#DERear":str(dercount)})
    # Serializing json
    json_object = json.dumps(js, indent = 4)

    filename, file_extension = os.path.splitext(os.path.basename(impath))

    # Writing to sample.json
    with open('./json/'+filename+".json", "w") as outfile:
        outfile.write(json_object)

export_json(r"C:/Corn_Production/CORNGROUPS/test.jpg")
#export_json(sys.argv[1])

# Figure 86,156 File "C:\Corn_Production\Production\fn_AI_process3.py", line 163, in <module>
#     export_json(r"C:/Corn_Production/CORNGROUPS/86.jpg")
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 131, in export_json
#     rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 127, in rbg
#     rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 69, in DRWAP_rate
#     if not (unique[isnum].isnumeric()):
# IndexError: index 4 is out of bounds for axis 0 with size 4


# 87,89 File "C:\Corn_Production\Production\fn_AI_process3.py", line 163, in <module>
#     export_json(r"C:/Corn_Production/CORNGROUPS/86.jpg")
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 131, in export_json
#     rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 127, in rbg
#     rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 69, in DRWAP_rate
#     if not (unique[isnum].isnumeric()):
# IndexError: index 3 is out of bounds for axis 0 with size 3


# 254 File "C:\Corn_Production\Production\fn_AI_process3.py", line 163, in <module>
#     export_json(r"C:/Corn_Production/CORNGROUPS/254.jpg")
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 131, in export_json
#     rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 127, in rbg
#     rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 69, in DRWAP_rate
#     if not (unique[isnum].isnumeric()):
# IndexError: index 2 is out of bounds for axis 0 with size 2


# 115,250,259,278,287,330 File "C:\Corn_Production\Production\fn_AI_process3.py", line 163, in <module>
#     export_json(r"C:/Corn_Production/CORNGROUPS/115.jpg")
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 131, in export_json
#     rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 127, in rbg
#     rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
#   File "C:\Corn_Production\Production\fn_AI_process3.py", line 69, in DRWAP_rate
#     if not (unique[isnum].isnumeric()):
# AttributeError: 'numpy.int32' object has no attribute 'isnumeric'


