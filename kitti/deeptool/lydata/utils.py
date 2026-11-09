from PIL import Image
import torch
import imageio
import numpy as np
import os
import cv2
import socket
import warnings

base_path = os.environ.get("DNINET_DATA_ROOT", "../data").rstrip("/") + "/"

def read_hole(hole_path, resize=None):
    # check if resize is None or function or float tuple or int tuple
    img = Image.open(hole_path)
    if resize is None:
        pass
    elif callable(resize):
        img = resize(img)
    elif isinstance(resize, tuple):
        if isinstance(resize[0], float) and isinstance(resize[1], float):
            img = img.resize((int(img.size[0] * resize[0]), int(img.size[1] * resize[1])))
        elif isinstance(resize[0], int) and isinstance(resize[1], int):
            img = img.resize((resize[0], resize[1]))
        else:
            raise ValueError("resize should be a function or a float tuple or a int tuple")
    
    hole = np.array(img).astype(np.float32)
    if np.max(hole) > 2.:
        hole = hole / 255.
    hole = (torch.from_numpy(hole)).unsqueeze(0)
    img.close()
    return hole

def read_rgb(rgb_path, resize=None):
    # check if resize is None or function or float tuple or int tuple
    img = Image.open(rgb_path)
    if resize is None:
        pass
    elif callable(resize):
        img = img.resize(resize(img.size))
    elif isinstance(resize, tuple):
        if isinstance(resize[0], float) and isinstance(resize[1], float):
            img = img.resize((int(img.size[0] * resize[0]), int(img.size[1] * resize[1])))
        elif isinstance(resize[0], int) and isinstance(resize[1], int):
            img = img.resize((resize[0], resize[1]))
        else:
            raise ValueError("resize should be a function or a float tuple or a int tuple")
    
    rgb = (np.array(img) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose(2, 0, 1))
    img.close()
    return rgb

def read_gt(gt_path, mode='sixteen', resize=None):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", category=UserWarning)
        # check if resize is None or function or float tuple or int tuple
        img = imageio.imread(gt_path)
        read_array = np.array(img)

        if mode == 'eight':
            gt = (read_array / 255.).astype(np.float32)
        elif mode == 'sixteen':
            gt = (read_array / 65535.).astype(np.float32)
        else:
            raise ValueError('read_gt_resize mode must be eight or sixteen')

        if resize is None:
            pass
        elif callable(resize):
            rec = resize(gt.shape)
            gt = cv2.resize(gt, (rec[1], rec[0]))
        elif isinstance(resize, tuple):
            if isinstance(resize[0], float) and isinstance(resize[1], float):
                gt = cv2.resize(gt, (int(gt.shape[1] * resize[0]), int(gt.shape[0] * resize[1])))
            elif isinstance(resize[0], int) and isinstance(resize[1], int):
                gt = cv2.resize(gt, (resize[1], resize[0]))
            else:
                raise ValueError("resize should be a function or a float tuple or a int tuple")

        gt = torch.from_numpy(gt).unsqueeze(0)
        del img
        return gt

# define a function search all file in a folder and return a list
def make_txt(path, *head):
    # path should not end with '/'
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            filelist = []
            path_dir = os.path.join(path, dir)
            for root_dir, dirs_dir, files_dir in os.walk(path_dir):
                for file in files_dir:
                    filelist.append(os.path.join(dir, file))
            # sort the list and write it line in line to a txt
            filelist.sort()
            record = open(path + "/" + dir + ".txt", "w")
            if len(head) > 0:
                record.write(head[0][dir] + "\n")
            for file in filelist:
                record.write(file + "\n")
            record.close()
    # print some information
    print("Make txt : " + path + " done!")
    print("Including these dirs : ")
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            print(dir)

# define a function to remove the '\n' of path
def pure_path(path):
    # check if there is '\n' in the end of path
    if path[-1] == '\n':
        return path[:-1]
    else:
        return path
    
if __name__ == "__main__":
    make_txt(base_path + "HRWSI/resize_64")
