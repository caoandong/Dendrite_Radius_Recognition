import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from sklearn.model_selection import train_test_split

def load_data(data_path="../data/simulations/data/data_radius.npy"):
    data = np.load(data_path)
    X = np.array(data[:,0])
    y = np.array(data[:,1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = np.stack(X_train)
    y_train = np.stack(y_train)
    return X_train, X_test, y_train, y_test

def get_mat_data(mat, start=None, diff=100, step=16, z_px=9, num_sep=16, num_fig=12, get_proj=True):
    mat_shape = mat.shape
    mid = int(mat_shape[-1]/2.0)
    if start is None:
        start = mid - diff
    ret_mat = []
    z_sep = int(1.0*step/num_sep)
    max_end = start + step*(num_fig-1) + z_sep*(num_sep-1)
    print('mat_shape: ', mat_shape,' max_end: ', max_end*z_px)
    assert(num_sep <= step)
    assert(max_end <= mat_shape[-1])
    depth_dict = {}
    for j in range(num_sep):
        mat_seg = []
        for i in range(num_fig):
            z_idx = start + step*i + z_sep*j
            depth = (mid - z_idx)*z_px
            mat_seg.append(mat[:,:,z_idx])
        ret_mat_tmp = mat_seg
        if get_proj:
            ret_mat_tmp = np.amax(np.array(mat_seg), axis=-1)
        ret_mat.append(ret_mat_tmp)
        depth_dict[str(depth)] = ret_mat_tmp
    return ret_mat, depth_dict

def read_mat_path(pathname="../data/simulations/", dz=0.45, max_num_file=1):
    mat_dict = {}
    num_file = 0
    for filename in os.listdir(pathname):
        num_file += 1
        if (max_num_file is not None) and (num_file > max_num_file):
            break
        if filename.endswith(".mat"):
          idx = filename.split('.')[0][8:]
          z_size = 90 + int(idx)*dz
          print("Filename: ", filename, " Radius: ", z_size)
          mat = scipy.io.loadmat(pathname+filename)
          mat_dict[str(z_size)] = np.array(mat['imagecg_new'])
    return mat_dict

def get_data_rad_depth(pathname="../data/simulations/", dz=0.45, max_num_file=10, z_px=9):
    data_rad = []
    data_depth = []
    mat_dict = read_mat_path(pathname=pathname, dz=dz, max_num_file=max_num_file)
    num_dict = len(mat_dict)
    for rad, mat in mat_dict.items():
        mat_data, depth_dict = get_mat_data(mat, z_px=z_px)
        for depth, mat in depth_dict.items():
            data_depth.append([mat, float(depth)])
        for j in range(len(mat_data)):
            data_rad.append([mat_data[j], float(rad)])
    return data_rad, data_depth

def generate_data(save_path="../data/simulations/data/", max_num_file=None):
    data_rad, data_depth = get_data_rad_depth(max_num_file=max_num_file)
    print("data_rad: ", len(data_rad), " data_depth: ", len(data_depth))
    np.save(save_path + 'data_radius.npy', data_rad)
    np.save(save_path + 'data_depth.npy', data_depth)
    print("Done.")

# generate_data()
# mat_dict = read_mat_path()
# num_dict = len(mat_dict)
# for key, mat in mat_dict.items():
#     mat_data, depth_dict = get_mat_data(mat)
#     num_data = len(mat_data)
#     print("Radius: ", key, " Image size: ", mat.shape, " Mat_data shape: ", num_data)
#     plt.figure(figsize=(10,10))
#     for i in range(num_data):
#         ax = plt.subplot(num_data/2, 2, i+1)
#         plt.imshow(mat_data[i])
# plt.show()
