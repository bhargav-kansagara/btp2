import numpy as np
import cv2
import math
import sys
np.set_printoptions(threshold=np.nan)

def show(image,name="image"):
	cv2.imshow(name,image)
	cv2.waitKey(0)

def slope(line=[]):
	if (line[0]!=line[2]):
		return math.degrees(math.atan(float(line[1]-line[3])/(line[0]-line[2]))) 
	else:
		return 90

def findlines(img,mul,x=10):
	thresh = img.copy()
	comp = np.zeros(img.shape,np.uint8)
	i=0
	while (i+mul<thresh.shape[0]):
		j=0
		while (j+mul<thresh.shape[1]):
			temp = []
			for i1 in range (i,i+mul):
				temp1 = []
				for j1 in range (j,j+mul):
					#print(i1,j1)
					temp1.append(thresh[i1][j1])
				temp.append(temp1)
			temp=np.array(temp)
			_,contours,h1 = cv2.findContours(temp,1,2)
			
			if (len(contours)>1):
				for i1 in range (i,i+mul):
					for j1 in range (j,j+mul):
						thresh[i1][j1]=0
						comp[i1][j1]=200
			j+=x
		i+=x
	return comp

def getComponents(image, components):
	CompImages = []
	for component in components:
		x,y,w,h = component
		if(h<w):
			img = np.zeros((h,w),np.uint8)
			for X in range(0,h):
				for Y in range(0,w):
					img[X][Y] = image[y+X][x+Y]
			show(img,"imgh")
			CompImages.append(img)
		else:
			img = np.zeros((w,h),np.uint8)
			for X in range(0,h):
				for Y in range(0,w):
					img[Y][X] = image[y+X][x+Y]
			show(img,"imgv")
			CompImages.append(img)
	return CompImages

file = "full/4.png"
# print(file)
i = cv2.imread(file,0)
show(i)
orig=i.copy()
img1=i.copy()
img2=i.copy()
#img=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
(wm,hm)=img1.shape
ret,thresh = cv2.threshold(img1,150,255,1)   ### skeletonization needs white on black so 1
ret,BW = cv2.threshold(img1,150,255,0)		# gives Black on white
# show(thresh,"before")
img=thresh.copy()
show(thresh,"cj")
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

final=thresh.copy()
sqsize = 20
shift = 5
comp=findlines(thresh,sqsize,shift)
show(comp,"comp")
components = []
wires = []
_,contours,h1 = cv2.findContours(comp,1 ,2)
for cont in contours:
	(x,y,w,h)= cv2.boundingRect(cont)
	if(w/h < 1.2 and w/h > 0.8):
		continue
	if(w<(sqsize*1.75) or h<(sqsize*1.75)):
		continue
	# print(x,y,w,h)
	x-=5
	y-=5
	w+=10
	h+=10
	components.append((x,y,w,h))
	cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.rectangle(final,(x,y),(x+w,y+h),0,-1)
	show(orig,"components")
print(components)

CompImages = getComponents(BW, components)

_,contours,h1 = cv2.findContours(final,1 ,2)
for cont in contours:
	(x,y,w,h)= cv2.boundingRect(cont)
	# print(x,y,w,h)
	cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
show(img2,"wires")

cv2.destroyAllWindows()
