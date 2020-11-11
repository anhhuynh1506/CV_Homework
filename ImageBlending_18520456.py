import cv2
import sys

def image_blending(back, fore, mask, alpha):

  img_b = cv2.imread(back)
  cv2.imshow('Original image', img_b)
  img_f = cv2.imread(fore)
  
  cv2.imshow('Effect image', img_f)
  img_m = cv2.imread(mask)
  
  img_b = img_b.astype('float64')
  img_f = img_f.astype('float64')
  img_m = img_m.astype('float64')

  img_f = cv2.resize(img_f,(img_b.shape[1],img_b.shape[0]))
  img_m = img_m / 100
  
  blend_img = (1 - alpha) * img_b + (alpha * img_f * img_m)

  cv2.imshow('Blending image', blend_img/255)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
    
def main():
  if len(sys.argv) != 5:
    print('usage: ImageBlending.py <ObjImage_path> <EffectImage_path> <MaskImage_path> <alpha>', file=sys.stderr)
    sys.exit(1)

  image_blending(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))

if __name__ == '__main__':
  main()