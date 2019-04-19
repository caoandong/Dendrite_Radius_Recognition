import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def plot_hist(data, start=0, end=500, num_step=50):
    df = pd.DataFrame({"data":data})
    bins= np.linspace(start, end, num=num_step)
    plt.hist(df.values, bins=bins, edgecolor="k")
    plt.xticks(bins)
    plt.show()

def load_data(data_path="../data/simulations/data/data_depth.npy"):
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
            ret_mat_tmp = np.amax(np.array(mat_seg), axis=0)
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
    print('mat_dict size: ', num_dict)
    for rad, mat in mat_dict.items():
        mat_data, depth_dict = get_mat_data(mat, z_px=z_px)
        for depth, mat in depth_dict.items():
            data_depth.append([mat, float(depth)])
        for j in range(len(mat_data)):
            data_rad.append([mat_data[j], float(rad)])
    return data_rad, data_depth

def save_data(data, save_path, file_name, len_data=None, split_step=100, delim=";"):
    file = None
    fname = None
    file_cnt = 0
    count = 0
    if len_data is None:
        len_data = len(data)
    for i in range(len_data):
        if count == 0:
            # Create a new file
            file_cnt += 1
            fname = save_path + file_name + "_" + str(file_cnt) + ".txt"
            file = open(fname, "w")
            print("Start writing new file: "+fname)
        elif count == split_step:
            file.close()
            print("Finished writing file: "+fname)
            count = 0
            continue
        count += 1
        img, label = data[i]
        img_txt = str(img.tolist())
        file.write(str(label)+delim+img_txt+'\n')
    if file is not None:
        file.close()
        print("Finished writing file: "+fname)

def split_test_train(data, test_prop=0.2):
    len_data = len(data)
    idx = shuffle(np.arange(len_data), random_state=0).tolist()
    split_idx = int(len_data*test_prop)
    test_idx = idx[:split_idx]
    train_idx = idx[split_idx:]
    test_data = [data[i] for i in test_idx]
    train_data = [data[i] for i in train_idx]
    return test_data, train_data, split_idx

def generate_data(save_path="../data/simulations/data/", max_num_file=None, test_prop=0.2):
    data_rad, data_depth  = get_data_rad_depth(max_num_file=max_num_file)
    rad_test, rad_train, rad_split_idx = split_test_train(data_rad)
    depth_test, depth_train, depth_split_idx = split_test_train(data_depth)
    len_rad = len(data_rad)
    len_rad_test = len(rad_test)
    len_rad_train = len(rad_train)
    len_depth = len(data_depth)
    len_depth_test = len(depth_test)
    len_depth_train = len(depth_train)
    print("rad test: ", len_rad_test, " rad train: ", len_rad_train, ' split idx: ', rad_split_idx)
    print("depth test: ", len_depth_test, " depth train: ", len_depth_train, ' split idx: ', depth_split_idx)
    np.save(save_path + "data_radius_test.npy", rad_test)
    # save_data(rad_test, save_path, "data_radius_test", len_rad_test)
    save_data(rad_train, save_path, "data_radius_train", len_rad_train)
    np.save(save_path + "data_depth_test.npy", depth_test)
    # save_data(depth_test, save_path, "data_depth_test", len_depth_test)
    save_data(depth_train, save_path, "data_depth_train", len_depth_train)
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
