import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import backend as K 
K.set_image_dim_ordering('th')
TEST_SET = ['2016Sentinel.png','2018GF.png']
predir=r'D:\Python\seg-data\data-GF\pre/'
image_size = 256

classes = [0. , 1]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,default='D:\Python\seg-data\data-GF\model\GF-test.h5',
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread(predir + path)

        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") 
        padding_img = img_to_array(padding_img)
        
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        print('src:',padding_img.shape,mask_whole.shape,padding_h,padding_w)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[:3,i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                _,ch,cw = crop.shape
                if (ch != 256 or cw != 256):
                    print ('invalid size!')
                    continue
                #print(set(crop.reshape(-1).tolist()))
                crop = np.expand_dims(crop, axis=0)
                #print 'crop:',crop.shape
                pred = model.predict_classes(crop,verbose=0)
                
                pred_prob = model.predict_proba(crop,verbose=1)
                
                pred = labelencoder.inverse_transform(pred[0])  
                #print (np.unique(pred))  
                pred = pred.reshape((256,256)).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        
        cv2.imwrite(predir+'pre'+path,mask_whole[0:h,0:w]*250)
        print ('pre:',mask_whole.shape,set(mask_whole.reshape(-1).tolist()))
    
from sklearn import preprocessing
import warnings
    
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    args = args_parse()
    predict(args)



