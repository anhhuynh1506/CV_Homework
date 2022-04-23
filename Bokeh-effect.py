import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('light_2.jpg',0)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()  

for i in 

blurImage = cv2.filter2D(img,-2,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blurImage),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
 
