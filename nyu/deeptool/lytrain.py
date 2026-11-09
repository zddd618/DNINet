import torch
import time
import os

# write a function to save train info result to txt which is a dict
def save_info(train_info, name="record", path="./saved/train/", is_origin=True):
    # check if the path esist, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
    txt_path = path + name + ".txt"

    # check if test_result is a dict or a list and save it
    if isinstance(train_info, dict):
        for key, value in train_info.items():
            with open(txt_path, 'a') as f:
                f.write(key + ':\n')
            # regard value as a dict and save it
            save_info(value, name=name, path=path, is_origin=False)
    elif isinstance(train_info, list):
        with open(txt_path, 'a') as f:
            for i in range(len(train_info)):
                f.write(str(train_info[i]) + '\n')
    elif isinstance(train_info, tuple):
        with open(txt_path, 'a') as f:
            for i in range(len(train_info)):
                f.write(str(train_info[i]) + '\n')
    else:
        with open(txt_path, 'a') as f:
            f.write(str(train_info) + '\n')

    # if is_origin is True, show some save information
    if is_origin:
        print("Train Info Saved: ", txt_path)
        print("Train Info Save Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# define a function to save essential information when the grad is too big with a threshold
def save_grad_info(model, grad_threshold=100, txt_path="./saved/train/grad_info.txt"):
    with open(txt_path, 'w') as f:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if param.grad.abs().max() > grad_threshold:
                    f.write(name + ':\n')
                    f.write(str(param.grad.abs().max().item()) + '\n')
    print("Grad Info Saved: ", txt_path)
    print("Grad Info Save Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# define a function to save the checkpoint at the place
def save_checkpoint(epoch, path, model, optimizer, *scaler):
    # check if the model is a nn.DataParallel model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # check if the scaler is empty or not
    if scaler:
        scaler = scaler[0]
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    # print information for saved checkpoint like save path and save time
    print("Checkpoint Saved: ", path)
    print("Checkpoint Save Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
