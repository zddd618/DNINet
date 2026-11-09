import cv2
import numpy as np
import torch
import os

from . import lydebug

def vision_gray(gray, name='gray.png', path='./saved/test/'):
    # check if path is valid if not create it
    if (not os.path.exists(path)):
        os.makedirs(path)
    full_path = path + name

    # check the type of x to be tensor or numpy array if tensor squeeze it and convert to numpy array
    if (type(gray) == torch.Tensor):
        # check if x requires grad if so detach it
        if (gray.requires_grad):
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (gray.is_cuda):
                x = gray.detach().cpu().clone()
            else:
                x = gray.detach().clone()
        else:
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (gray.is_cuda):
                x = gray.cpu().clone()
            else:
                x = gray.clone()
        if (len(x.shape) == 4):
            x = x[0, :, :, :]
        x = x.squeeze()
        x = x.numpy()
    elif (type(gray) == np.ndarray):
        x = np.squeeze(gray)
        
    # check the shape of x to be 3D and 2D if not return error
    if (len(x.shape) != 2):
        print("Message: vision_gray got more than 2 dimensions")
        if (len(x.shape) == 3):
            # judge the type of x to be float or int
            if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
                # convert to uint8
                x = (x * 255).astype(np.uint8)
            for i in range(x.shape[0]):
                # save it to path with cv2
                cv2.imwrite(path + str(i) + name, np.squeeze(x[i]))
            return 0
        else:
            print("Error: vision_gray got invalid shape")
            return 0
    
    # judge the type of x to be float or int
    if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
        # convert to uint8
        x = (x * 255).astype(np.uint8)

    # save it to path with cv2
    cv2.imwrite(full_path, x)
    # print some info mation about save the image like save .. to ..
    print("Message: vision_gray save image to " + full_path)

def vision_binary(binary, name='binary.png', path='./saved/test/'):
    # check if path is valid if not create it
    if (not os.path.exists(path)):
        os.makedirs(path)
    full_path = path + name
    
    # check the type of x to be tensor or numpy array if tensor squeeze it and convert to numpy array
    if (type(binary) == torch.Tensor):
        if (binary.requires_grad):
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (binary.is_cuda):
                x = binary.detach().cpu().clone()
            else:
                x = binary.detach().clone()
        else:
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (binary.is_cuda):
                x = binary.cpu().clone()
            else:
                x = binary.clone()
        if (len(x.shape) == 4):
            x = x[0, :, :, :]
        x = x.squeeze()
        x = x.numpy()
    elif (type(binary) == np.ndarray):
        x = np.squeeze(binary)
    
    # check the shape of x to be 3D and 2D if not return error
    if (len(x.shape) != 2):
        print("Message: vision_binary got more than 2 dimensions")
        if (len(x.shape) == 3):
            # judge the type of x to be float or int
            if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
                # convert to uint8
                x = (x * 255).astype(np.uint8)
            # convert to binary by threshold and x is a uint8 array use cv2
            x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]
            for i in range(x.shape[0]):
                # save it to path with cv2
                cv2.imwrite(path + str(i) + name, np.squeeze(x[i]))
            return 0
        else:
            print("Error: vision_binary got invalid shape")
            return 0
    
    # judge the type of x to be float or int
    if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
        # convert to uint8
        x = (x * 255).astype(np.uint8)

    # convert to binary by threshold and x is a uint8 array use cv2
    x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]

    # save it to path with cv2
    cv2.imwrite(full_path, x)
    # print some info mation about save the image like save .. to ..
    print("Message: vision_binary save image to " + full_path)

def vision_rgb(rgb, name='rgb.png', path='./saved/test/'):
    # check if path is valid if not create it
    if (not os.path.exists(path)):
        os.makedirs(path)
    full_path = path + name
    
    # check the type of x to be tensor or numpy array if tensor squeeze it and convert to numpy array
    if (type(rgb) == torch.Tensor):
        if (rgb.requires_grad):
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (rgb.is_cuda):
                x = rgb.detach().cpu().clone()
            else:
                x = rgb.detach().clone()
        else:
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (rgb.is_cuda):
                x = rgb.cpu().clone()
            else:
                x = rgb.clone()
        x = x.squeeze()
        x = x.numpy()
    elif (type(rgb) == np.ndarray):
        x = np.squeeze(rgb)

    # check the shape of x to be 3D and 2D if not return error
    if (len(x.shape) != 3):
        print("Error: vision_rgb got invalid shape")
        return 0
    
    # judge the type of x to be float or int
    if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
        # convert to uint8
        x = (x * 255).astype(np.uint8)

    # if 3 is the first dimension change it to last dimension
    if (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    # change the order of channels to BGR
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    # save it to path with cv2
    cv2.imwrite(full_path, x)
    # print some info mation about save the image like save .. to ..
    print("Message: vision_rgb save image to " + full_path)

def vision_bgr(bgr, name='bgr.png', path='./saved/test/'):
    # check if path is valid if not create it
    if (not os.path.exists(path)):
        os.makedirs(path)
    full_path = path + name
    
    # check the type of x to be tensor or numpy array if tensor squeeze it and convert to numpy array
    if (type(bgr) == torch.Tensor):
        if (bgr.requires_grad):
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (bgr.is_cuda):
                x = bgr.detach().cpu().clone()
            else:
                x = bgr.detach().clone()
        else:
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (bgr.is_cuda):
                x = bgr.cpu().clone()
            else:
                x = bgr.clone()
        x = x.squeeze()
        x = x.numpy()
    elif (type(bgr) == np.ndarray):
        x = np.squeeze(bgr)

    # check the shape of x to be 3D and 2D if not return error
    if (len(x.shape) != 3):
        print("Error: vision_bgr got invalid shape")
        return 0
    
    # judge the type of x to be float or int
    if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
        # convert to uint8
        x = (x * 255).astype(np.uint8)

    # if 3 is the first dimension change it to last dimension
    if (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    # save it to path with cv2
    cv2.imwrite(full_path, x)
    # print some info mation about save the image like save .. to ..
    print("Message: vision_bgr save image to " + full_path)

def vision_color(depth, name='color.png', path='./saved/test/'):
    # check if path is valid if not create it
    if (not os.path.exists(path)):
        os.makedirs(path)
    full_path = path + name

    # check the type of x to be tensor or numpy array if tensor squeeze it and convert to numpy array
    if (type(depth) == torch.Tensor):
        # check if x requires grad if so detach it
        if (depth.requires_grad):
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (depth.is_cuda):
                x = depth.detach().cpu().clone()
            else:
                x = depth.detach().clone()
        else:
            # check if x is cpu or gpu tensor if gpu convert it to cpu
            if (depth.is_cuda):
                x = depth.cpu().clone()
            else:
                x = depth.clone()
        if (len(x.shape) == 4):
            x = x[0, :, :, :]
        x = x.squeeze()
        x = x.numpy()
    elif (type(depth) == np.ndarray):
        x = np.squeeze(depth)
        
    # check the shape of x to be 3D and 2D if not return error
    if (len(x.shape) != 2):
        print("Message: vision_color got more than 2 dimensions")
        if (len(x.shape) == 3):
            # judge the type of x to be float or int
            if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
                # convert to uint8
                x = (x * 255.).astype(np.uint8)
            for i in range(x.shape[0]):
                # save it to path with cv2
                img_color = cv2.applyColorMap(np.squeeze(x[i]), cv2.COLORMAP_PLASMA) # 应用plasma颜色映射
                cv2.imwrite(path + str(i) + name, img_color)
            return 0
        else:
            print("Error: vision_color got invalid shape")
            return 0
    
    # judge the type of x to be float or int
    if (x.dtype == np.float32 or x.dtype == np.float16 or x.dtype == np.float64):
        # convert to uint8
        x = (x * 255.).astype(np.uint8)

    # save it to path with cv2
    # img_color = cv2.applyColorMap(x, cv2.COLORMAP_PLASMA) # 应用plasma颜色映射
    img_color = cv2.applyColorMap(x, cv2.COLORMAP_TURBO)
    
    cv2.imwrite(full_path, img_color)
    # print some info mation about save the image like save .. to ..
    print("Message: vision_color save image to " + full_path)
