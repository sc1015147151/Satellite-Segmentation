
import pandas as pd
import numpy as np
import csv
import re
import cv2 
from matplotlib import pyplot as plt
data_csv=csv.reader(open('2016.csv','r'))
line1=next(data_csv)
line2=next(data_csv)
line3=next(data_csv)
line4=next(data_csv)
line5=next(data_csv)
line6=next(data_csv)
line7=next(data_csv)
line8=next(data_csv)
line9=next(data_csv)
print(line1,line2,line3,line8,line9)
roi_data=[]
for row in data_csv: 
        roi_data.append(row)
roi_data = [[float(x) for x in row] for row in roi_data]
roiDataFrame= pd.DataFrame(roi_data,  columns = line8)
width=int(re.split(r'\s+', line2[0])[3])
height=int(re.split(r'\s+', line2[0])[5])
backGroundImg=np.zeros((height,width))
for i in range(roiDataFrame.shape[0]):   
    backGroundImg[int(roiDataFrame[line8[1]][i])][int(roiDataFrame[line8[0]][i])]=1
plt.imshow(backGroundImg, cmap = 'gray')
cv2.imwrite('2016labelV.png', backGroundImg*255)
cv2.imwrite('2016label.png', cv2.merge([backGroundImg,backGroundImg,backGroundImg]))





