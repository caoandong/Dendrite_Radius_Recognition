import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv
import os

img_dir = os.getcwd()+'/data/img1/0.tif'
csv_dir = os.getcwd()+'/data/img1/curves/'
img_scale = 221.87
# Load image
img_arr = Image.open(img_dir)
img = np.array(img_arr)
img = np.stack((img,)*3, axis=-1)
img = np.uint8(img)
print(img.shape)
w = img.shape[0]
h = img.shape[1]
scale_x = float(w)/img_scale
scale_y = float(h)/img_scale
soma_pos = np.array((83,92))
soma_rad = 10

cont_dict = {}
cont_pts = []
cont_len = {}
cont_id_max = 0
with open(csv_dir + 'contours.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    prev_id = -1
    for row in csv_reader:
        # print(row)
        cur_id = row['Contour ID']
        if cur_id != prev_id:
            if prev_id != -1:
                cont_dict[prev_id] = cont_pts
                cont_len[prev_id] = row['Length']
            prev_id = cur_id
            cont_pts = []
        cont_pts.append([int(float(row['X'])), int(float(row['Y']))])
        if cont_id_max < int(cur_id):
            cont_id_max = int(cur_id)
print('Number of lines: ', len(cont_dict))



close_cont_id = []
for cont_id, cont_pts in cont_dict.items():
    p0 = cont_pts[0]
    p1 = cont_pts[-1]
    d0 = np.sqrt((p0[0]-soma_pos[0])**2+(p0[1]-soma_pos[1])**2)
    d1 = np.sqrt((p1[0]-soma_pos[0])**2+(p1[1]-soma_pos[1])**2)
    if d0 < soma_rad*10:
        close_cont_id.append(cont_id)
    elif d1 < soma_rad:
        cont_dict[cont_id] = cont_pts[::-1]
        close_cont_id.append(cont_id)

for id in close_cont_id:
    cont_pts = cont_dict[id]
    color = int(int(id)/cont_id_max * 255)
    for i in range(len(cont_pts)-1):
        p0 = (int(cont_pts[i][0]*scale_x), int(cont_pts[i][1]*scale_y))
        p1 = (int(cont_pts[i+1][0]*scale_x), int(cont_pts[i+1][1]*scale_y))
        cv2.line(img, p0, p1, (color,255,0), 1)
img = cv2.resize(img, (1080, 1080))
cv2.imshow('img', img)
cv2.waitKey(0)
