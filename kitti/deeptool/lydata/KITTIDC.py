import os
import shutil
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
if __name__ == "__main__":
    import utils
else:
    from . import utils

method_list = {"intrinsics":"binary",
               "img":"rgb",
               "gt":"depth",
               "raw":"depth",
               "gt_ori":"depth",
               "seg":"binary",
               "relative":"depth"}

num_kind = {"val": 1000,
            "val_0.1": 1000,
            "val_0.05": 1000,
            "val_0.01": 1000,
            "val_0.005": 1000,
            "val_0.001": 1000}

def reorder():
    print("Remain to add reorder code, but don't add seems OK.")

def check_states():
    for kind in ["val_rel", "train-full"]:
        base_path = utils.base_path + "KITTIDC/" + kind
        dirs = os.listdir(base_path)
        # if ("img" in dirs) and ("gt" in dirs) and ("raw" in dirs):
        #     if(len(os.walk(os.path.join(base_path, "gt")).__next__()[2]) != num_kind[kind]):
        #         print("The dataset KITTIDC/" + kind + " the gt is broken, please check it.")
        #         print("The number of PNG files in " + kind + " is " + str(len(os.walk(os.path.join(base_path, "gt")).__next__()[2])) + ".")    
        #         raise ValueError("Broken dataset.")
        #     if(len(os.walk(os.path.join(base_path, "img")).__next__()[2]) != num_kind[kind]):
        #         print("The dataset KITTIDC/" + kind + " the gt is broken, please check it.")
        #         print("The number of PNG files in " + kind + " is " + str(len(os.walk(os.path.join(base_path, "img")).__next__()[2])) + ".")    
        #         raise ValueError("Broken dataset.")
        #     if(len(os.walk(os.path.join(base_path, "gt_ori")).__next__()[2]) != num_kind[kind]):
        #         print("The dataset KITTIDC/" + kind + " the gt is broken, please check it.")
        #         print("The number of PNG files in " + kind + " is " + str(len(os.walk(os.path.join(base_path, "gt_ori")).__next__()[2])) + ".")    
        #         raise ValueError("Broken dataset.")
        # else:
        #     print("The dataset KITTIDC/" + kind + " is not reordered.")
        #     raise ValueError("Broken dataset.")
        
        for each_path in dirs:
            if not each_path.endswith(".txt"):
                if((each_path + ".txt") not in dirs):
                    print("The dataset KITTIDC/" + kind + " is broken, the file " + each_path + ".txt is missing.")
                    print("Rebuild the dataset KITTIDC's txts ...")
                    utils.make_txt(base_path, method_list)
                    print("Rebuild finished.")
                    break
    print("Check_state and the dataset KITTIDC is OK.")

class DATASET:
    def __init__(self, kind) -> None:
        super().__init__()
        # build list
        self.base_path = utils.base_path + "KITTIDC/" + kind
        # check if base_path end with '/'
        if self.base_path[-1] != '/':
            self.base_path += '/'

        self.data_dict = {}

        # search all txt file in the base_path
        for root, dirs, files in os.walk(self.base_path):
            if os.path.normpath(root) == os.path.normpath(self.base_path):
                for file in files:
                    if file[-4:] == ".txt":
                        record = open(os.path.join(root, file), 'r')
                        lines = record.readlines()
                        tmp_dict = {}
                        tmp_dict["method"] = utils.pure_path(lines[0])
                        tmp_list = []
                        for line in lines[1:]:
                            tmp_list.append(utils.pure_path(line))
                        record.close()
                        tmp_dict["list"] = tmp_list
                        self.data_dict[file[:-4]] = tmp_dict

    # define len and getitem
    def measure(self):
        return len(self.data_dict["img"]["list"])

    def select(self, index, chosen_list = ["img", "gt"]):
        result = {}
        for chosen in chosen_list:
            tmp_path = self.base_path + self.data_dict[chosen]["list"][index]
            result["path"] = os.path.basename(tmp_path)
            result["full_path"] = tmp_path
            if self.data_dict[chosen]["method"] == "rgb":
                result[chosen] = utils.read_rgb(tmp_path)
            elif self.data_dict[chosen]["method"] == "depth":
                result[chosen] = utils.read_gt(tmp_path)
            elif self.data_dict[chosen]["method"] == "binary":
                result[chosen] = utils.read_hole(tmp_path)
            else:
                raise ValueError("Unknown method")

        return result

def build_val():
    base_path = utils.base_path + "KITTIDC/val"
    # check if base_path end with '/'
    if base_path[-1] != '/':
        base_path += '/'

    data_dict = {}

    # search all txt file in the base_path
    for root, dirs, files in os.walk(base_path):
        if os.path.normpath(root) == os.path.normpath(base_path):
            for file in files:
                if file[-4:] == ".txt":
                    record = open(os.path.join(root, file), 'r')
                    lines = record.readlines()
                    tmp_dict = {}
                    tmp_dict["method"] = utils.pure_path(lines[0])
                    tmp_list = []
                    for line in lines[1:]:
                        tmp_list.append(utils.pure_path(line))
                    record.close()
                    tmp_dict["list"] = tmp_list
                    data_dict[file[:-4]] = tmp_dict

    name_pro = {"val_0.1": 0.1, "val_0.05": 0.05, "val_0.01": 0.01, "val_0.005": 0.005, "val_0.001": 0.001}
    for name in ["val_0.1", "val_0.05", "val_0.01", "val_0.005", "val_0.001"]:
        img_path = utils.base_path + "KITTIDC/" + name + "/img"
        gt_path = utils.base_path + "KITTIDC/" + name + "/gt"
        gt_ori_path = utils.base_path + "KITTIDC/" + name + "/gt_ori"
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(gt_ori_path, exist_ok=True)
        loop = tqdm(range(num_kind["val"]), desc="Building " + name)
        for i in loop:
            img_path = base_path + data_dict["img"]["list"][i]
            img = Image.open(img_path)
            img = np.array(img)
            gt_ori_path = base_path + data_dict["gt_ori"]["list"][i]
            gt_ori = imageio.imread(gt_ori_path)
            gt_ori = np.array(gt_ori)
            gt_path = base_path + data_dict["gt"]["list"][i]
            gt = imageio.imread(gt_path)
            gt = np.array(gt)

            height, width = img.shape[0], img.shape[1]
            height_goal = (height // 64) * 64
            width_goal = (width // 64) * 64
            start = (height - height_goal, width//2 - width_goal//2)

            img = img[start[0]:start[0]+height_goal, start[1]:start[1]+width_goal, :]
            gt = gt[start[0]:start[0]+height_goal, start[1]:start[1]+width_goal]
            gt_ori = gt_ori[start[0]:start[0]+height_goal, start[1]:start[1]+width_goal]

            num_sample = int(height_goal*width_goal*name_pro[name])

            gt_flat = gt.flatten()
            idx_nnz = np.nonzero(gt_flat >= 1)[0]

            num_idx = len(idx_nnz)
            if num_idx > num_sample:
                idx_sample = np.random.permutation(num_idx)[:num_sample]
                idx_nnz = idx_nnz[idx_sample]

            mask = np.zeros(height_goal * width_goal)
            mask[idx_nnz] = 1.0
            mask = mask.reshape((height_goal , width_goal))

            gt_sp = gt * mask.astype(gt.dtype)

            img_save_path = img_path.replace("val", name)
            gt_save_path = gt_path.replace("val", name)
            gt_ori_save_path = gt_ori_path.replace("val", name)

            imageio.imwrite(gt_save_path, gt_sp)
            Image.fromarray(img).save(img_save_path)
            imageio.imwrite(gt_ori_save_path, gt_ori)

            loop.set_postfix({"img": img_save_path, "gt": gt_save_path})
        
        utils.make_txt(utils.base_path + "KITTIDC/" + name, method_list)

if __name__ == "__main__":
    reorder()
    check_states()
    build_val()

