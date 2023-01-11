#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf 
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:
    print("Please install GPU version of TF")
import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.model import MaskRCNN
import numpy as np
import argparse
import random
import cv2
import os
from matplotlib import pyplot
from PIL import Image
from tensorflow.keras.models import load_model
from os import listdir
from sklearn.cluster import KMeans
from yellowbrick.cluster import kelbow_visualizer
#get_ipython().run_line_magic('matplotlib', 'inline')
#ตั้งค่าให้ code ใช้ CUDA_VISIBLE_DEVICES ของ GPU ในการ Train ข้อมูล
os.environ["CUDA_DEVICE_ORDER"]="0000:01:00.0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
os.environ["OMP_NUM_THREADS"] = '1'


# In[ ]:


class myMaskRCNNConfig(Config):
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
     # kangaroo + BG
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 131
    
    # Learning rate
    LEARNING_RATE=0.006
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10





# In[ ]:


def imgandcoor(im):
    img=im
    imgs=np.any(img,axis=2)
    vertical=np.where(np.any(imgs,axis=1))[0]
    horizon=np.where(np.any(imgs,axis=0))[0]
    imgs=imgs[vertical[0]:vertical[-1],horizon[0]:horizon[-1]]
    img=img[vertical[0]:vertical[-1],horizon[0]:horizon[-1]]
    coordinate=np.where(imgs==True)
    coordinate[0][:]=coordinate[0][::-1]
    return img,coordinate
    
def vlmean(coordinate1,coordinate0):
    X=coordinate1
    Y=coordinate0
    meanY=np.mean(coordinate0)
    meanX=np.mean(coordinate1)
    X=X-meanX
    Y=Y-meanY
    absZ_alt = X**2 + Y**2
    reZ_alt = (X**4 - 6*X**2*Y**2 + Y**4)/absZ_alt
    imZ_alt = (4*X**3*Y - 4*X*Y**3)/absZ_alt
    phi_alt = np.arctan2(np.sum(imZ_alt), np.sum(reZ_alt))/4
    return np.degrees(phi_alt)
    

def rotation(img,angle):
    p1=np.array((0,0))
    p2=np.array((img.shape[0],img.shape[1]))
    diagonal=(int)(np.ceil(np.linalg.norm(p1-p2)))
    zero=np.zeros((diagonal,diagonal,3),dtype='uint8')
    y1=(int)(np.round(zero.shape[0]/2)-np.round(img.shape[0]/2))
    x1=(int)(np.round(zero.shape[1]/2)-np.round(img.shape[1]/2))
    zero[y1:y1+img.shape[0],x1:x1+img.shape[1]]=+img
    image=Image.fromarray(zero)
    rotated = image.rotate(angle)
    img=np.array(rotated)
    imgs=np.any(img,axis=2)
    vertical=np.where(np.any(imgs,axis=1))[0]
    horizon=np.where(np.any(imgs,axis=0))[0]
    img=img[vertical[0]:vertical[-1],horizon[0]:horizon[-1]]
    imgs=imgs[vertical[0]:vertical[-1],horizon[0]:horizon[-1]]
    coordinate=np.where(imgs==True)
    coordinate[0][:]=coordinate[0][::-1]
    return img,coordinate

def imrotated(im):
    img,coordinate=imgandcoor(im)
    phi=0
    angle=vlmean(coordinate[1],coordinate[0])
    correlation=np.corrcoef(coordinate[1],coordinate[0])
    #print(angle)
    #print(img.shape[0]/2,img.shape[1]/2)
    if img.shape[0]/(img.shape[1])<=1:
        if angle<0:
            angle-=-90
        else:
            angle-=90
            
        if angle*correlation[1][0]>0:
                if angle<0:
                    angle-=-90
                else:
                    angle-=90
                
                if round(abs(angle)) ==0:
                    if angle<0:
                        angle-=-90
                        #angle2-=-90
                    else:
                        angle-=90
                        #angle2-=90
    
    if round(abs(angle))!=0 and angle*correlation[1][0]>0:
            if angle<0:
                angle-=90
            else:
                angle-=-90
        
    #print(angle)
    img,coordinate=rotation(img,-angle)
    return img


# In[ ]:


def get_rcnn(model_dir,model_path,config):     
    #โหลด โมเดลที่เราพึ่งเทรนไปล่าสุด
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.keras_model.load_weights(model_path, by_name=True)
    der=""
    for name in os.walk(model_dir):
        der=name[0]
    #os.rmdir(der)
    return model


# In[ ]:


def pre_processing(path,model):
    dir = './ear/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    
    image=cv2.imread(path) 
    results = model.detect([image], verbose=0)
    r = results[0]
    print("r ====> {r}".format(r=r))
    print("r['rois'].shape[0] ====> {rd}".format(rd=r['rois'].shape[0]))
    
    #Move form 241
    ims=[]
    
    if r['rois'].shape[0]==0 :
        return image,r,ims,0,0
    if r['rois'].shape[0]==1:
        return  image,r,ims,0,1
    hcoord=r['rois'][:,0]
    lcoord=r['rois'][:,2]
    centerY=r['rois'][:,0]+r['rois'][:,2]
    centerX=r['rois'][:,1]+r['rois'][:,3]
    centerX=np.array(centerX)
    centerY=np.array(centerY)
    imaxY=np.argmax(lcoord)
    iminY=np.argmin(hcoord)
    sortmarks=[]
    sortrois=[]
    iteration=10
    print(hcoord,lcoord,centerY,centerX,imaxY,iminY)
    if r['rois'].shape[0]<=8:
        iteration=r['rois'].shape[0]+1
    if (centerY[imaxY]/2-hcoord[imaxY])/2+hcoord[imaxY]>lcoord[iminY]:
        #zero=np.zeros((centerY.shape))
        K=np.column_stack((centerY,hcoord,lcoord))
        K=np.array(K)
        #print(K)
        model = KMeans()
        visualizer = kelbow_visualizer(model, K,k=(1, iteration),timings=False,show=False)
        #visualizer.fit(K)
        cluster=visualizer.elbow_value_
        if cluster is None or cluster==1 :
            cluster=r['rois'].shape[0]

        model = KMeans(n_clusters=cluster).fit(K)

        #print(np.std(model.cluster_centers_[:,1]),np.std(centerY))
    
        
        sortrows=np.argsort(model.cluster_centers_[:,1])
        predict=np.zeros((model.labels_ .shape[0]),dtype='uint8')
        for i in range(sortrows.shape[0]):
            where_i=np.where(model.labels_==sortrows[i])
            where_i=np.array(where_i)
            np.put(predict,where_i[0],i)
        rows=[]
        for i in range(cluster):
            rows.append(np.where(predict==i))
        for k in range(cluster):
            column=[centerX[j] for j in rows[k][0]]
            column=np.array(column)
            column=np.argsort(column)
            for i in range(rows[k][0].shape[0]):
                sortmarks.append(r['masks'][:,:,rows[k][0][column[i]]])
                sortrois.append(r['rois'][rows[k][0][column[i]],:])
    else:
        cluster=1
        column=np.argsort(centerX)
        for i in range(column.shape[0]):
            sortmarks.append(r['masks'][:,:,column[i]])
            sortrois.append(r['rois'][column[i],:])
    sortmarks=np.array(sortmarks)
    sortrois=np.array(sortrois)
    
    for i in range(sortrois.shape[0]):  
        x=sortrois[i]
        im=image[x[0]:x[2],x[1]:x[3]].copy()
        rr=sortmarks[i,:,:]
        rr=rr.astype('uint8')
        rr=rr[x[0]:x[2],x[1]:x[3]]
        im[rr==0]=0
        im=imrotated(im) #หมุนภาพ
        ims.append(im)
        print("ims ====> {ims}".format(ims=ims))
        cv2.imwrite('./ear/'+str(i+1)+".jpg",im) #ถ้าหมุนภาพ cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return image,r,ims,cluster,sortrois  #input_image, mask-RCNN_prediction, croping_images, n_cluster


