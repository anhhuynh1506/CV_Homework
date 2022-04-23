import cv2 
import numpy as np
test = cv2.imread("tesstas.jpg")
d = cv2.imread("bg.jpg")
test_hsv = cv2.cvtColor(test,cv2.COLOR_BGR2HSV)


def HSV_ColorRangeDetect(test_hsv):
  u = 0
  xz = 0
  u2 = 0
  u3 = 0
  xz2 = 0
  xz3 = 0
  for i in range (test_hsv.shape[0]):
    for j in range(300,400):
      u = u + test_hsv[i][j][0]
      u2 = u2 + test_hsv[i][j][1]
      u3 = u3 + test_hsv[i][j][2]
  u = ( u / (i*j))
  u2 = ( u2 /(i*j) )
  u3 = ( u3 / (i*j) )
  for i in range (test_hsv.shape[0]):
    for j in range(300,400):
      xz = xz + (test_hsv[i][j][0]- u)*(test_hsv[i][j][0]- u)
      xz2 = xz2 + (test_hsv[i][j][1]- u2)*(test_hsv[i][j][1]- u2)
      xz3 = xz3 + (test_hsv[i][j][2]- u3)*(test_hsv[i][j][2]- u3)
  xz = xz / (i*j)
  xz = ( np.sqrt(xz) )
  xz2 = xz2 / (i*j)
  xz2 = ( np.sqrt(xz2) )
  xz3 = xz3 / (i*j)
  xz3 = np.sqrt(xz3) 
  return u, u2,u3,xz,xz2,xz3
u = (HSV_ColorRangeDetect(test_hsv)[0])
u2 = (HSV_ColorRangeDetect(test_hsv)[1])
u3 = (HSV_ColorRangeDetect(test_hsv)[2])
xz = (HSV_ColorRangeDetect(test_hsv)[3])
xz2 = (HSV_ColorRangeDetect(test_hsv)[4])
xz3 = (HSV_ColorRangeDetect(test_hsv)[5])


lower_green = np.array([np.abs(u- 2*xz),np.abs(u2- 2*xz2) ,np.abs(u3- 2*xz3)])
upper_green= np.array([u+2*xz,u2+2*xz2 ,u3+2*xz3])
mask = cv2.inRange(test_hsv,lower_green,upper_green)
mask = cv2.bitwise_not(mask)
kernel = np.ones((3,3), np.uint8)
erote  =cv2.erode(mask,kernel,1)
eroted = cv2.erode(erote,kernel,1)
dilated = cv2.dilate(eroted,kernel,1)
res = cv2.bitwise_and(test,test, mask= eroted)
ab = (test.shape[1],test.shape[0])
d = cv2.resize(d,ab)
HSV = np.where(res == 0,d,res)
cv2.imshow("HSV image",HSV)

#RGB COLOR DETECT


g = test[:,:,1]
def RGB_RangeColorDetect(test):
  g1 = 0
  for i in range (test_hsv.shape[0]):
    for j in range(46):
      g1 = g1+ g[i][j]
  g1 = g1 / (280*45)
  x3 = 0
  for i in range (test_hsv.shape[0]):
    for j in range(45):
      x3 = x3+ (g[i][j]-g1)**2
  
  x3 = x3 / float(280*45)
  x3 = np.sqrt(x3)
  return g1,x3

g1 = RGB_RangeColorDetect(test)[0]
x3 = RGB_RangeColorDetect(test)[1]


d = test.copy()
mask = np.zeros(test.shape)
for i in range(test.shape[0]):
  for j in range(test.shape[1]):
    if g[i][j] < np.abs(g1-3*x3) or g[i][j] > (g1+3*x3):
      mask[i][j] = d[i][j]   
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
d = cv2.imread("bg.jpg")
d = cv2.resize(d,ab)
RGB = np.where(mask == 0,d,mask)
cv2.imshow("RGB image",RGB)






