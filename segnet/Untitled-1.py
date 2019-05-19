

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
dir='D:/Python/seg-data/data_MB/test/src'


data=np.load(dir+'/0.npy')




cv.imshow('123',np.floor((data[0]/(data[0].max()))*255))
cv.waitKey()
#plt.imshow(data[2])




