import os
import shutil
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import h5py
from tqdm import tqdm
if __name__ == "__main__":
    import utils
else:
    from . import utils

method_list = {"img":"rgb",
               "gt":"depth",
               "gt_ori":"depth",
               "seg":"binary",
               "h5_set":"h5",
               "relative":"depth"}

num_kind = {"val": 654,
            "val_0.1": 654,
            "val_0.05": 654,
            "val_0.01": 654,
            "val_0.005": 654,
            "val_0.001": 654,
            "val_rel": 654}

def reorder():
    print("No need to reorder because all h5 file.")
    print("Just check_states and will be OK.")

def check_states():
    for kind in ["val", "val_0.1", "val_0.05", "val_0.01", "val_0.005", "val_0.001", "val_rel"]:
        base_path = utils.base_path + "NYUBIG/" + kind
        dirs = os.listdir(base_path)
        if ("h5_set" in dirs):
            if(len(os.walk(os.path.join(base_path, "h5_set")).__next__()[2]) != num_kind[kind]):
                print("The dataset NYUBIG/" + kind + " the h5_set is broken, please check it.")
                print("The number of h5 files in " + kind + " is " + str(len(os.walk(os.path.join(base_path, "h5_set")).__next__()[2])) + ".")  
                raise ValueError("Broken dataset.")
        # else:
        #     print("The dataset NYUBIG/" + kind + " is not reordered.")
        #     raise ValueError("Broken dataset.")
        
        for each_path in dirs:
            if not each_path.endswith(".txt"):
                if((each_path + ".txt") not in dirs):
                    print("The dataset NYUBIG/" + kind + " is broken, the file " + each_path + ".txt is missing.")
                    print("Rebuild the dataset NYUBIG's txts ...")
                    utils.make_txt(base_path, method_list)
                    print("Rebuild finished.")
                    break
    print("Check_state and the dataset NYUBIG is OK.")

class DATASET:
    def __init__(self, kind) -> None:
        super().__init__()
        # build list
        self.base_path = utils.base_path + "NYUBIG/" + kind
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
        if "h5_set" in self.data_dict.keys():
            return len(self.data_dict["h5_set"]["list"])
        else:
            return len(self.data_dict["img"]["list"])

    def select(self, index, chosen_list = ["img", "gt"]):
        result = {}
        for chosen in chosen_list:
            tmp_path = self.base_path + self.data_dict[chosen]["list"][index]
            if self.data_dict[chosen]["method"] == "rgb":
                result[chosen] = utils.read_rgb(tmp_path)
            elif self.data_dict[chosen]["method"] == "depth":
                result[chosen] = utils.read_gt(tmp_path)
            elif self.data_dict[chosen]["method"] == "binary":
                result[chosen] = utils.read_hole(tmp_path)
            elif self.data_dict[chosen]["method"] == "h5":
                result[chosen] = utils.read_h5(tmp_path)
            else:
                raise ValueError("Unknown method")

        return result

def build_val():
    base_path = utils.base_path + "NYUBIG/val"
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
        img_path = utils.base_path + "NYUBIG/" + name + "/img"
        gt_path = utils.base_path + "NYUBIG/" + name + "/gt"
        gt_ori_path = utils.base_path + "NYUBIG/"+ name + "/gt_ori"
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(gt_ori_path, exist_ok=True)
        loop = tqdm(range(num_kind["val"]), desc="Building " + name)
        for i in loop:
            h5_path = base_path + data_dict["h5_set"]["list"][i]
            f = h5py.File(h5_path, 'r')

            img = f['rgb'][:].transpose(1, 2, 0)
            gt = f['depth'][:]

            height, width = img.shape[0], img.shape[1]
            height_goal = (height // 64) * 64
            width_goal = (width // 64) * 64
            start = (height//2 - height_goal//2, width//2 - width_goal//2)

            img = img[start[0]:start[0]+height_goal, start[1]:start[1]+width_goal, :]
            gt = gt[start[0]:start[0]+height_goal, start[1]:start[1]+width_goal] / 10.0

            num_sample = int(height_goal*width_goal*name_pro[name])

            gt_flat = gt.flatten()
            idx_nnz = np.nonzero(gt_flat > 0.000001)[0]

            num_idx = len(idx_nnz)
            idx_sample = np.random.permutation(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample]

            mask = np.zeros(height_goal * width_goal)
            mask[idx_nnz] = 1.0
            mask = mask.reshape((height_goal, width_goal))

            gt_sp = gt * mask.astype(gt.dtype)
            gt_sp = (gt_sp*65535.).astype(np.uint16)
            gt = (gt*65535.).astype(np.uint16)

            img_save_path = img_path + "/" + str(i) + ".png"
            gt_save_path = gt_path + "/" + str(i) + ".png"
            gt_ori_save_path = gt_ori_path + "/" + str(i) + ".png"

            imageio.imwrite(gt_save_path, gt_sp)
            Image.fromarray(img).save(img_save_path)
            imageio.imwrite(gt_ori_save_path, gt)

            loop.set_postfix({"img": img_save_path, "gt": gt_save_path})
                    
        utils.make_txt(utils.base_path + "NYUBIG/" + name, method_list)


if __name__ == "__main__":
    reorder()
    check_states()
    build_val()
