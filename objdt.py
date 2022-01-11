import cv2 

img = cv2.imread('leo.jpg')

#Empty matrix
classNames= []

#Dataset
classfile = 'coco.names'

#Adding datas into a single matrix
with open(classfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

configpaths='ssd_mobilenet_v1_large_coco_2020_01_14.pbtxt'
weightspath='frozen_interface_graph.pb'

#Boundary box allocation 
net=cv2.dnn_DetectionModel(weightspath,configpaths)

net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5,127.5,127.5)
net.setInputSwapRB(True)




cv2.imshow("Output",img)
cv2.waitKey(0)
