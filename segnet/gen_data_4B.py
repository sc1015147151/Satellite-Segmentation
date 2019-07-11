#coding=utf-8
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import gdal
from skimage import exposure
import matplotlib.pyplot as plt
img_w = 256  
img_h = 256 
data_dir=r'D:/Python/seg-data/data_MB/'

image_sets = ['2018']
def tiff2array(tif_data):
    x_width  = tif_data.RasterXSize    #栅格矩阵的列数
    x_height = tif_data.RasterYSize
    array_data=tif_data.ReadAsArray(0,0,x_width,x_height)
    array_data=np.floor((array_data/(array_data.max()))*255)
    return array_data,x_width,x_height
def changeImgDataShape(originalShapeData):
    data=originalShapeData
    #print(data.shape)
    changedShapeData=np.append(np.append(np.append(data[3,:,:][:,:,np.newaxis],data[2,:,:][:,:,np.newaxis],axis=2),data[1,:,:][:,:,np.newaxis],axis=2),data[0,:,:][:,:,np.newaxis],axis=2)
    #print(changedShapeData.shape)
    return changedShapeData
def data_augment(image,label,channel=4):    
    channnels=np.zeros(image.shape)
    w=image.shape[1]
    center=((w-1)/2,(w-1)/2)
    randomInt2=random.randint(0,3)
    randomInt1=random.randint(-1,1)
    randomInt3=random.randint(-1,1)
    randomInt4=random.choice([0.5,0.75,1,1.5,2])
    for i in range(channel):
        #print(channnels[i].shape)
        #plt.imshow(channnels[i])
        #plt.show()
        #print(channnels[i])
        channnels[i]=image[i]
        channnels[i]=cv2.flip(channnels[i],randomInt1)
        channnels[i]=cv2.flip(channnels[i],randomInt3) 
        parameters = cv2.getRotationMatrix2D(center,randomInt2*90, 1)
        channnels[i] = cv2.warpAffine(channnels[i], parameters, (w, w))
        channnels[i]= exposure.adjust_gamma(channnels[i], randomInt4)
        #print(channnels[i])
        #plt.imshow(channnels[i])
        #plt.show()
    label=cv2.flip(label,randomInt1)
    label=cv2.flip(label,randomInt3)
    label=cv2.warpAffine(label, parameters, (w, w))
    #plt.imshow(label)
    #plt.show()
    return channnels,label
    return image,label
def data_augment1(xb,yb):
    xb,yb=flip(xb,yb)

    return xb,yb
def creat_dataset(image_num = 1500, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        data_tif=gdal.Open(data_dir + image_sets[i]+'.tif')
        im_data,x_width,x_height=tiff2array(data_tif)
        #im_data=changeImgDataShape(im_data)
       #####################################
        label_img = cv2.imread(data_dir+image_sets[i]+'.png' ,cv2.IMREAD_GRAYSCALE)  
        while count < image_each:
            random_width = random.randint(0, x_width - img_w - 1)
            random_height = random.randint(0, x_height - img_h - 1)
            

            src_roi = im_data[:,random_height: random_height + img_h, random_width: random_width + img_w]
            
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
                
                label_roi_band0=src_roi[:,:,0]
            src_roi=changeImgDataShape(src_roi)    
            if (label_roi_band0[label_roi_band0==0].size<10000):
                visualize = np.zeros((256,256)).astype(np.uint8)
                visualize = label_roi*250
                black_num = 0
                for column in label_roi:
                    for pixel in  column:
                        if (pixel == 0):
                            black_num=black_num+1
                if (black_num<256*256*0.5):   
                    cv2.imwrite((data_dir+'labelV/%d.png' % g_count),visualize)
                    np.save(data_dir+'src/%d' % g_count,src_roi)
                    cv2.imwrite((data_dir+'label/%d.png' % g_count),label_roi)
                    
                    
                    count += 1 
                    g_count += 1
if __name__=='__main__':  
    creat_dataset(150,'augment')
