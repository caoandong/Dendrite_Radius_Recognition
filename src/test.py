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
img_color = np.stack((img,)*3, axis=-1)
img = np.uint8(img)
img_color = np.uint8(img_color)
print(img.shape)
w = img.shape[0]
h = img.shape[1]
scale_x = float(w)/img_scale
scale_y = float(h)/img_scale
soma_pos = np.array((83,92))
soma_rad = 50

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
        cont_pts.append([float(row['X']), float(row['Y'])])
        if cont_id_max < int(cur_id):
            cont_id_max = int(cur_id)
print('Number of lines: ', len(cont_dict))



close_cont_id = []
for cont_id, cont_pts in cont_dict.items():
    p0 = cont_pts[0]
    p1 = cont_pts[-1]
    d0 = np.sqrt((p0[0]-soma_pos[0])**2+(p0[1]-soma_pos[1])**2)
    d1 = np.sqrt((p1[0]-soma_pos[0])**2+(p1[1]-soma_pos[1])**2)
    if d0 < soma_rad:
        close_cont_id.append(cont_id)
    elif d1 < soma_rad:
        cont_dict[cont_id] = cont_pts[::-1]
        close_cont_id.append(cont_id)

def plot_radius(id, img, img_color, step=10, bin=10, show_plot=0):
    cont_pts = cont_dict[id]
    color = int(int(id)/cont_id_max * 255)
    cnt = 0
    num_seg = int((len(cont_pts)-1.0)/step)
    # print(len(cont_pts))
    # print(int((len(cont_pts)-1.0)/step))
    for i in range(num_seg):
        p0 = (cont_pts[step*i][0]*scale_x, cont_pts[step*i][1]*scale_y)
        p1 = (cont_pts[step*(i+1)][0]*scale_x, cont_pts[step*(i+1)][1]*scale_y)
        normal = np.array(p1) - np.array(p0)
        normal = 2*np.array([-1*normal[1], normal[0]])
        n0 = np.array(p0) + normal
        n1 = np.array(p0) - normal
        dir = n1 - n0
        dist = np.linalg.norm(dir)
        n0 = (int(n0[0]), int(n0[1]))
        n1 = (int(n1[0]), int(n1[1]))
        dist = np.sqrt((n0[0] - n1[0])**2 + (n0[1] - n1[1])**2)
        cv2.line(img_color, n0, n1, (color,255,0), 1)
        if show_plot != 0:
        # if cnt <= show_plot and show_plot != 0:
            intensity = []
            plt.clf()
            for j in range(int(dist)):
                p = np.array(n0) + j*1.0/int(dist)*dir
                x = int(p[0])
                y = int(p[1])
                intensity.append(img[y:y+1,x:x+1][0])
                cv2.circle(img_color, (x,y), 1, (255,0,255))
            # print(intensity)
            plt.plot(intensity)
            plt.savefig('data/img1/trace/test_trace_%s.png'%str(i))
            cnt += 1
    return img, img_color

# for id in close_cont_id:
#     cont_pts = cont_dict[id]
#     color = int(int(id)/cont_id_max * 255)
#     for i in range(len(cont_pts)-1):
#         p0 = (int(cont_pts[i][0]*scale_x), int(cont_pts[i][1]*scale_y))
#         p1 = (int(cont_pts[i+1][0]*scale_x), int(cont_pts[i+1][1]*scale_y))
#         cv2.line(img, p0, p1, (color,255,0), 1)
cnt = 0
# for id in close_cont_id:
#     if cnt <= 1:
max_id = 0
max_len = 0
for i in close_cont_id:
    len_tmp = len(cont_dict[i])
    if max_len < len_tmp:
        max_len = len_tmp
        max_id = i
img, img_color = plot_radius(max_id, img, img_color, show_plot=10)
        # cnt += 1
img_color = cv2.resize(img_color, (1080, 1080))
cv2.imshow('img', img_color)
cv2.waitKey(0)
