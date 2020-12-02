import cv2 
import numpy as np 
import math
import sys

def HoughTransform(img, rho=1, theta=np.pi/180, threshold=200):
    H, W = img.shape[:2]
    diagonal_length = int(math.sqrt(H*H + W*W))    
    num_r = int(diagonal_length / rho)
    num_theta = int(np.pi / theta)
    edge_matrix = np.zeros([2 * num_r + 1, num_theta])
    idx	= np.squeeze(cv2.findNonZero(img)) 
    range_theta = np.arange(0, np.pi, theta)
    theta_matrix = np.stack((np.cos(np.copy(range_theta)), np.sin(np.copy(range_theta))), axis=-1)
    vote_matrix = np.dot(idx, np.transpose(theta_matrix))
    for vr in range(vote_matrix.shape[0]):
        for vc in range(vote_matrix.shape[1]):
            rho_pos = int(round(vote_matrix[vr, vc]))+num_r
            edge_matrix[rho_pos, vc] += 1    
    line_idx = np.where(edge_matrix > threshold)
    
    r_values = list(line_idx[0])
    r_values = [r - num_r for r in r_values]
    t_values = list(line_idx[1])
    t_values = [t/180.0*np.pi for t in t_values]
    
    line_idx = list(zip(r_values, t_values))
    line_idx = [[li] for li in line_idx]
    return line_idx
def out_img(img_path):
    img = cv2.imread(img_path) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    edges_result = cv2.Canny(gray,200,100,apertureSize = 3) 
    lines = HoughTransform(edges_result)
    for line in lines[:5]:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)   
    
    out_name = img_path.split('.jpg')[0] + '_result.jpg'
    cv2.imwrite(out_name,img)

def main():
    if len(sys.argv) != 2:
        print('usage: hough_transform.py <image_path>', file=sys.stderr)
        sys.exit(1)

    out_img(sys.argv[1])

if __name__ == '__main__':
    main()