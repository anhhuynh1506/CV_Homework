import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Histogram(image):

  #Step 0: Convert RGB color space to HSV space
  img = cv2.imread(image,1)
  conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(conv_img) # hue, saturation, value(brightness)
  plt.hist(V.ravel(),256,[0,256])

  # Step 1: Perform histogram equalization on V
  pixel = np.zeros(256)
  v_arr = V.reshape(-1)
  for v in v_arr:
      pixel[v] += 1
  cdf = 0
  cdf_min = np.min(pixel[pixel>0])
  npixel = len(v_arr)
  output_value = V.copy()

  #Step 3: Convert H,S,V* to RGB color space
  mapping = {}
  for value in range(0,256):
      # Calculate CDF(value)
      cdf += pixel[value]
      # Calculate h(value)
      h_v = np.round((cdf - cdf_min)/(npixel - cdf_min)*(256-1))
      mapping[value] = h_v
      output_value[V==value] = h_v
  plt.hist(output_value.ravel(),256,[0,256])

  #Step 4: Show Equalized Histogram Result 
  conv_img = cv2.merge([H,S,output_value])
  Img_Af_Equal = cv2.cvtColor(conv_img, cv2.COLOR_HSV2BGR)
  cv2.imshow('Original image', img)
  cv2.imshow('Equalized image', Img_Af_Equal)
  cv2.waitKey()
  cv2.destroyAllWindows()
 
def main():
  if len(sys.argv) != 2:
    print('usage: Histogram.py <Image_path>', file=sys.stderr)
    sys.exit(1)
  print(sys.argv[1])
  Histogram(sys.argv[1])

if __name__ == '__main__':
    main()