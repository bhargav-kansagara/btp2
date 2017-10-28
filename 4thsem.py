import numpy as np
import cv2
import math
import matplotlib.path as mpltPath
import xml.etree.cElementTree as ET
import urllib
from PIL import Image
import sys
print(sys.argv)
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


def show(image,name="image"):
	cv2.imshow(name,image)
	cv2.waitKey(0)
def slope(line=[]):
	if (line[0]!=line[2]):
		return math.degrees(math.atan(float(line[1]-line[3])/(line[0]-line[2]))) 
	else:
		return 90


def DiffLines(lines=[[]]):
	diffLines=[]
	diffLines.append([[lines[0][0][0],lines[0][0][1],lines[0][0][2],lines[0][0][3]]])
	for i1 in range (1,len(lines)):
		i=lines[i1]
		x1=i[0][0]
		x2=i[0][2]
		y1=i[0][1]
		y2=i[0][3]
		theta1=slope(i[0])
		flag=0
		for j in diffLines:
			theta2=slope(j[0])
			if (abs(theta2-theta1)<10 and ((abs(slope([j[0][0],j[0][1],x1,y1])-theta1)<10 and (((x2-j[0][0])**2+(y2-j[0][1])**2)**0.5)<400) or (abs(slope([j[0][0],j[0][1],x2,y2])-theta1)<10 and (((x1-j[0][0])**2+(y1-j[0][1])**2)**0.5)<400))):
				j.append([i[0][0],i[0][1],i[0][2],i[0][3]])
				flag=1
				break
		if flag==0:
			diffLines.append([[i[0][0],i[0][1],i[0][2],i[0][3]]])
	return diffLines

def findlines(image,mul):
        thresh=image
        length = int(thresh.shape[0]/mul)
        width = int(thresh.shape[1]/mul)
        print(thresh.shape)
        #show(thresh,"thresh")
        for i in range (0,length):
                for j in range (0,width):
                        temp = []
                        for i1 in range (mul*i,mul*(i+1)):
                                temp1= []
                                for j1 in range (mul*j,mul*(j+1)):
                                        #print(i1,j1)
                                        temp1.append(thresh[i1][j1])
                                temp.append(temp1)
                        temp=np.array(temp)	
                        #print(temp)
                        img=temp.copy()
                        _,contours,h1 = cv2.findContours(temp,1,2)
                        #print(len(contours))
                        # print(contours)
                        # if (len(contours)>0):
                                #cv2.imshow("temp",img)
                                #cv2.waitKey(0)
                        if (len(contours)>1):
                                #print(img)
                                for i1 in range (mul*i,mul*(i+1)):
                                        for j1 in range (mul*j,mul*(j+1)):
                                                thresh[i1][j1]=0
        #show(thresh,"thresh")
        return thresh


file = "36.png"
print(file)
i = cv2.imread(file,0)
show(i)
orig=i.copy()
img1=i.copy()
img2=i.copy()
#img=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
(wm,hm)=img1.shape
ret,thresh = cv2.threshold(img1,150,255,1)   ### skeletonization requires black image so 1
show(thresh,"before")
lines=[[[]]]
#ret,thresh = cv2.threshold(img1,200,255,1) 
#cv2.imshow("ther",thresh)
#cv2.waitKey(0)
img=thresh

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
skel = np.zeros(img.shape,np.uint8)
size = np.size(img)
while(done==0):
	eroded = cv2.erode(img,element)
	temp = cv2.dilate(eroded,element)
	temp = cv2.subtract(img,temp)
	skel = cv2.bitwise_or(skel,temp)
	img = eroded.copy()
	zeros = size - cv2.countNonZero(img)
	if zeros==size:
		done = True

thresh=skel
#show(skel)
ret,thresh = cv2.threshold(thresh,100,255,0)

show(thresh,"thresh")
#print(thresh)
##cv2.imshow("ther",thresh)
##cv2.waitKey(0)
thresh1=thresh.copy()
final=thresh.copy()
img1=findlines(thresh,50)
img2=findlines(thresh1,100)
show(img1,"img1")
show(img2,"img2")
for i in range (0,img1.shape[0]):
        for j in range (0,img1.shape[1]):
                if (img1[i][j]==0 or img2[i][j]==0):
                        final[i][j]=0
                else:
                        final[i][j]=255
show(final,name="final")
#show(img1,name="img1")	
#show(thresh)
_,contours,h1 = cv2.findContours(thresh,1 ,2)
#print(contours)
# _,contours,h1 = cv2.findContours(thresh,1,2)

for cont in contours:
	print("approx")
	approx = cv2.approxPolyDP(cont,0.01*cv2.arcLength(cont,True),True)
	print(approx)
	(x,y,w,h)= cv2.boundingRect(approx)
	print(x,y,w,h)
	cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255),2)
	show(orig,"i")


cv2.destroyAllWindows()

# lines = cv2.HoughLinesP(thresh,10,np.pi/20,200)
# print lines

# diff=DiffLines(lines)
# for line in diff:
# 	cv2.line(img2,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,0,255),2)

# cv2.imshow("output",img2)
# cv2.waitKey(0)
