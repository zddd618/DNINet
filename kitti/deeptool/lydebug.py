import torch
import numpy as np
from torchsummary import summary
import os
import inspect

# Write a function to check tensor's basic information
def check_tensor(tensor):
    # print check tensor in which function
    print("Check tensor in function: ", inspect.stack()[1][3])
    print("Tensor Type: ", tensor.type())
    print("Tensor Shape: ", tensor.shape)
    print("Tensor Device: ", tensor.device)
    # print max min mean
    print("Tensor Max: ", torch.max(tensor))
    print("Tensor Min: ", torch.min(tensor))
    print("Tensor Mean: ", torch.mean(tensor))

# Write a function to check ndarray's basic information
def check_array(ndarray):
    # print check ndarray in which function
    print("Check ndarray in function: ", inspect.stack()[1][3])
    print("ndarray Type: ", type(ndarray))
    print("ndarray Shape: ", ndarray.shape)
    # print max min mean
    print("ndarray Max: ", np.max(ndarray))
    print("ndarray Min: ", np.min(ndarray))
    print("ndarray Mean: ", np.mean(ndarray))

# Write a function to check list's basic information
def check_list(list):
    # print check list in which function
    print("Check list in function: ", inspect.stack()[1][3])
    print("List Length: ", len(list))
    print("List Content: ", list)

# Write a function to check list's basic information
def check_tuple(tuple):
    # print check tuple in which function
    print("Check tuple in function: ", inspect.stack()[1][3])
    print("Tuple Length: ", len(tuple))
    print("Tuple Content: ", tuple)

# Write a function to check dict's basic information
def check_dict(dict):
    # print check dict in which function
    print("Check dict in function: ", inspect.stack()[1][3])
    print("Dict Length: ", len(dict.keys()))
    # print keys and values
    print("Dict Keys: ", dict.keys())
    print("Dict Values: ", dict.values())

# Write a function to check torch model's basic information
def check_model(model, input_size):
    # print check model in which function
    print("Check model in function: ", inspect.stack()[1][3])
    if input_size:
        summary(model, input_size)
    else:
        # print number of parameters and trainable parameters
        print("Model Parameters: ", sum([param.nelement() for param in model.parameters()]))
        print("Model Trainable Parameters: ", sum([param.nelement() for param in model.parameters() if param.requires_grad]))
        print("Model Structure: ", model)

# Write a function to check torch dataset's basic information
def check_dataset(dataset, indices=0):
    # print check dataset in which function
    print("Check dataset in function: ", inspect.stack()[1][3])
    # print dataset length and the selected indices of dataset content and content type corresponding
    print("Dataset Length: ", len(dataset))
    # print dataset content type looply if the dataset has multiple types of content
    if isinstance(dataset[indices], list) or isinstance(dataset[indices], tuple):
        for i in range(len(dataset[indices])):
            print("Dataset Content Type " + str(i) + " : ", type(dataset[indices][i]))
            if 'Tensor' in type(dataset[indices][i]).__name__:
                check_tensor(dataset[indices][i])
            elif 'ndarray' in type(dataset[indices][i]).__name__:
                check_array(dataset[indices][i])
            elif 'list' in type(dataset[indices][i]).__name__:
                check_list(dataset[indices][i])
            elif 'dict' in type(dataset[indices][i]).__name__:
                check_dict(dataset[indices][i])
            elif 'tuple' in type(dataset[indices][i]).__name__:
                check_tuple(dataset[indices][i])
            else:
                print(dataset[indices][i])
    else:
        print("Dataset Content Type: ", type(dataset[indices]))

def check_directory(path):
    # Check if the directory exists, if not, create it, and print some information
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created: {}'.format(path))
    else:
        print('Directory already exists: {}'.format(path))
