#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import gdal
import warnings
warnings.filterwarnings("ignore")
img_w = 256  
img_h = 256 


image_sets = ['test.tif']
def tiff2array(tif_data):
    x_width  = tif_data.RasterXSize    #栅格矩阵的列数
    x_height = tif_data.RasterYSize
    array_data=tif_data.ReadAsArray(0,0,x_width,x_height)
    return array_data,x_width,x_height
def changeImgDataShape(originalShapeData):
    data=originalShapeData
    changedShapeData=np.append(np.append(np.append(data[3,:,:][:,:,np.newaxis],data[2,:,:][:,:,np.newaxis],axis=2),data[1,:,:][:,:,np.newaxis],axis=2),data[0,:,:][:,:,np.newaxis],axis=2)
    return changedShapeData
def data_augment(xb,yb):

    return xb,yb
def creat_dataset(image_num = 15000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        data_tif=gdal.Open('./data_MB/' + image_sets[i])
        im_data,x_width,x_height=tiff2array(data_tif)
        im_data=changeImgDataShape(im_data)
       
        label_img = cv2.imread('./data_MB/test_label.png' ,cv2.IMREAD_GRAYSCALE)  
        while count < image_each:
            random_width = random.randint(0, x_width - img_w - 1)
            random_height = random.randint(0, x_height - img_h - 1)

            src_roi = im_data[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
                
                src_roi_band0=src_roi[0]
                
            
            if (src_roi_band0[src_roi_band0==0].size<60000):
                visualize = np.zeros((256,256)).astype(np.uint8)
                visualize = label_roi *250
                black_num = 0
                for column in src_roi_band0:
                    for pixel in  column:
                        
                        if (pixel == 0):
                            black_num=black_num+1
                if (black_num<256*256*0.65):
                    cv2.imwrite(('./data_MB/test/labelV/%d.png' % g_count),visualize)

                    np.save('./data_MB/test/src/%d' % g_count,src_roi)
                    cv2.imwrite(('./data_MB/test/label/%d.png' % g_count),label_roi)
                    
                    
                    count += 1 
                    g_count += 1
def gen_dataset(image_num = 15000, mode = 'original',stride=256):
    image_sets=['2016']

    for k in image_sets:
        data_tif=gdal.Open('D:\Python\seg-data/data_MB/' + k+'.tif')
        label_img = cv2.imread('D:\Python\seg-data/data_MB/' + k+'.png' ,cv2.IMREAD_GRAYSCALE)  

        count=0
        im_data,tif_width,tif_height=tiff2array(data_tif)
        #cv2.imshow(' 0 ',im_data[1])
        cv2.waitKey(0)
        im_data=changeImgDataShape(im_data)
        h,w,_ = im_data.shape
        padding_h = (tif_height//stride + 1) * stride 

        padding_w = (tif_width//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,_))
        padding_label = np.zeros((padding_h,padding_w))
        padding_img[0:h,0:w,:] = im_data[:,:,:]
        padding_label[0:h,0:w] = label_img[:,:]
        b1,b2,b3,b4=cv2.split(padding_img)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+256,j*stride:j*stride+256,:]
                crop_label=padding_label[i*stride:i*stride+256,j*stride:j*stride+256]
                
                b1,b2,b3,b4=cv2.split(crop)
                if (np.sum(b1==0)!=(256*256)): 
                    count=count+1
                    cv2.imwrite(f'D:\Python\seg-data/data_MB/label/{count}.png',crop_label )
                    cv2.imwrite(f'D:\Python\seg-data/data_MB/visual/{count}.png',crop_label*255 )

                    np.save(f'D:\Python\seg-data/data_MB/img/{count}.npy',crop)
                ch,cw,_ = crop.shape
                print(crop.shape)
                if (ch != 256 or cw != 256):
                    print ('invalid size!')
                    continue
                print (im_data.shape,tif_width,tif_height)    
if __name__=='__main__':  
    warnings.filterwarnings("ignore")
    gen_dataset(100,'augment')
