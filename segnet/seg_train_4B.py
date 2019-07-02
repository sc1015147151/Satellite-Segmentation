
#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
from keras import backend as K 
K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import gdal
seed = 7  
np.random.seed(seed)  
  
#data_shape = 360*480  
img_w = 256  
img_h = 256  
#有一个为背景  
n_label =2
  
classes = [0. ,  1]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  



def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img



def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    filepath ='..\..\Python\seg-data\data_MB/test/'  
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set
# data for training  



# data for training  
def generateData(batch_size,data=[]): 
    print ('generateData...')
    filepath ='..\..\Python\seg-data\data_MB/test/'  
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = np.load((filepath + 'src/' + url))
            #img = np.load( filepath + 'src/' + url ) 
            train_data.append(img)  
            label = load_img((filepath + 'label/' + url).replace('npy','png'), grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,-1))  
 
            train_label.append(label)  
            if batch % batch_size==0: 
        
                train_data = np.array(train_data)  
                
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 




# data for validation 
def generateValidData(batch_size,data=[]):  
    print ('generateValidData...')
    filepath ='..\..\Python\seg-data\data_MB/test/'  
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1 
            img = np.load((filepath + 'src/' + url))
            #img = np.load( filepath + 'src/' + url ) 
            valid_data.append(img)  
            label = load_img((filepath + 'label/' + url).replace('npy','png'), grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,-1))  
 
            valid_label.append(label) 
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  



def SegNet():  
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    nClasses=2
    input_height=256
    input_width=256
    img_input = Input(shape=( 4,input_height, input_width))

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x, mask_1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x , mask_2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x, mask_3 = MaxPoolingWithArgmax2D(name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    x, mask_4 = MaxPoolingWithArgmax2D(name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x, mask_5 = MaxPoolingWithArgmax2D(name='block5_pool')(x)

    Vgg_streamlined=Model(inputs=img_input,outputs=x)

    # 加载vgg16的预训练权重
    #Vgg_streamlined.load_weights(r"E:\Code\PycharmProjects\keras-segmentation\data\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # 解码层
    unpool_1 = MaxUnpooling2D()([x, mask_5])
    y = Conv2D(512, (3,3), padding="same")(unpool_1)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_2 = MaxUnpooling2D()([y, mask_4])
    y = Conv2D(512, (3, 3), padding="same")(unpool_2)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_3 = MaxUnpooling2D()([y, mask_3])
    y = Conv2D(256, (3, 3), padding="same")(unpool_3)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_4 = MaxUnpooling2D()([y, mask_2])
    y = Conv2D(128, (3, 3), padding="same")(unpool_4)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(64, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_5 = MaxUnpooling2D()([y, mask_1])
    y = Conv2D(64, (3, 3), padding="same")(unpool_5)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(nClasses, (1, 1), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Reshape((-1, nClasses))(y)
    y = Activation("softmax")(y)

    model=Model(inputs=img_input,outputs=y)
    return model  


def train(): 
    EPOCHS = 1000
    BS = 4
    model = SegNet() 
    
    modelcheck = ModelCheckpoint('..\..\Python\seg-data/model/4Bands-'+time.strftime(f'%Y-%m-%d-%a-%H-%M-%S',time.localtime(time.time()))+'.h5',monitor='val_acc',save_best_only=True,mode='max')  
    
    callable = [modelcheck]  
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb,train_numb//BS)  
    print ("the number of val data is",valid_numb,valid_numb//BS)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)  

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")



def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment",help="using data augment or not",
                    action="store_true", default=False)
    ap.add_argument("-m", "--model", required=True,default='model/'+time.strftime(f'%Y-%m-%d %a %H:%M:%S',time.localtime(time.time()))+'.h5'
                    ,help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args



train()  
 


