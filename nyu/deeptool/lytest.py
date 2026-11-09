import time
import torch
import numpy as np
from . import lyprocess
import random
import os

alpha = 1e-8

def RMSE(result, goal):
    # check if result and goal are both torch.Tensor or numpy.ndarray
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()
    dif = result - goal
    dif = dif * dif
    dif[goal == 0.] = 0
    return np.sqrt(np.sum(dif) / np.count_nonzero(goal))

def MAE(result, goal):
    # check if result and goal are both torch.Tensor or numpy.ndarray
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()
    dif = np.abs(result - goal)
    dif[goal == 0.] = 0
    return np.sum(dif) / np.count_nonzero(goal)

def SRMSE(result, goal):
    # check if result and goal are both torch.Tensor or numpy.ndarray
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()
    dif = (lyprocess.mean_var_norm(result) - lyprocess.mean_var_norm(goal))
    dif = dif * dif
    dif[goal == 0.] = 0
    return np.sqrt(np.sum(dif) / np.count_nonzero(goal))

def ARE(result, goal):
    # check if result and goal are both torch.Tensor or numpy.ndarray
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()
    dif = (result - goal) / (goal + alpha)
    dif[goal == 0.] = 0
    return np.sum(np.abs(dif)) / np.count_nonzero(goal)

def RMSE_tensor(result, goal):
    # 确保 result 和 goal 都是 torch.Tensor
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    if not isinstance(goal, torch.Tensor):
        goal = torch.tensor(goal)
    mask = goal > 0.0001
    dif = result[mask] - goal[mask]
    dif_sqr = torch.pow(dif, 2)
    return torch.sqrt(torch.sum(dif_sqr) / (mask.sum() + alpha))

def MAE_tensor(result, goal):
    # 确保 result 和 goal 都是 torch.Tensor
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    if not isinstance(goal, torch.Tensor):
        goal = torch.tensor(goal)
    mask = goal > 0.0001
    dif = result[mask] - goal[mask]
    dif_abs = torch.abs(dif)
    return torch.sum(dif_abs) / (mask.sum() + alpha)

def SRMSE_tensor(result, goal):
    # 确保 result 和 goal 都是 torch.Tensor
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    if not isinstance(goal, torch.Tensor):
        goal = torch.tensor(goal)
    # 假设 lyprocess.mean_var_norm 已经被修改为接受 torch.Tensor
    dif = (lyprocess.mean_var_norm(result) - lyprocess.mean_var_norm(goal))
    dif = dif * dif
    dif[goal == 0.] = 0
    return torch.sqrt(torch.sum(dif) / torch.count_nonzero(goal))

def ARE_tensor(result, goal):
    # 确保 result 和 goal 都是 torch.Tensor
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    if not isinstance(goal, torch.Tensor):
        goal = torch.tensor(goal)
    mask = goal > 0.0001
    dif = result[mask] - goal[mask]
    dif = (result - goal) / (goal + alpha)
    dif[goal == 0.] = 0
    return torch.sum(torch.abs(dif)) / torch.count_nonzero(goal)

def ODE(result, goal):
    # check if result and goal are both torch.Tensor or numpy.ndarray
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()

    tao = 0.05
    loss = 0

    result_np = result[0, 0, :, :]
    goal_np = goal[0, 0, :, :]
    height, width = result_np.shape

    node1 = []
    node2 = []
    for i in range(height):
        for j in range(width):
            if(goal_np[i][j] > 0):
                node1.append([i, j])
                node2.append([i, j])
    random.shuffle(node1)
    random.shuffle(node2)
    node1 = random.sample(node1, int(len(node1) / 2))
    node2 = random.sample(node2, int(len(node2) / 2))

    for i in range(len(node1)):
        result_record = 0
        goal_record = 0

        rate = result_np[node1[i][0]][node1[i][1]] / (result_np[node2[i][0]][node2[i][1]] + alpha)
        if rate > 1 + tao:
            result_record = 1
        else:
            if rate < 1 / (1 + tao):
                    result_record = -1

        rate = goal_np[node1[i][0]][node1[i][1]] / (goal_np[node2[i][0]][node2[i][1]] + alpha)
        if rate > 1 + tao:
            goal_record = 1
        else:
            if rate < 1 / (1 + tao):
                goal_record = -1

        if goal_record != result_record:
            loss += 1

    return loss / (len(node1) + alpha)

# write a function to save test result to txt which is a dict
def save_result(test_result, name="result", path="./saved/test/", is_origin=True, time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())):
    # check if the path esist, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
    txt_path = path + name + "_" + time_stamp + ".txt"

    # check if test_result is a dict or a list and save it
    if isinstance(test_result, dict):
        for key, value in test_result.items():
            with open(txt_path, 'a') as f:
                f.write(key + ':\n')
            # regard value as a dict and save it
            save_result(value, name=name, path=path, is_origin=False, time_stamp=time_stamp)
    elif isinstance(test_result, list):
        with open(txt_path, 'a') as f:
            for i in range(len(test_result)):
                f.write(str(test_result[i]) + '\n')
    elif isinstance(test_result, tuple):
        with open(txt_path, 'a') as f:
            for i in range(len(test_result)):
                f.write(str(test_result[i]) + '\n')
    else:
        with open(txt_path, 'a') as f:
            f.write(str(test_result) + '\n')

    # if is_origin is True, show some save information
    if is_origin:
        print("Test Result Saved: ", txt_path)
        print("Test Result Save Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))       
