from torch.utils.data import Dataset
from PIL import Image
import os
import shutil

class MyData(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.path = os.path.join(data_dir, label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.data_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.label_dir
        return img, lable

    def __len__(self):
        return len(self.img_path_list)


def creat_label(root_dir, file_lists, file_kind, file_content):
    path = os.path.join(root_dir, file_content+'_label')
    if not os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    file_names = [os.path.splitext(f)[0] for f in file_lists]
    for file_name in file_names:
        file_dir = os.path.join(path, file_name+file_kind)
        f = open(file_dir, 'w')
        f.writelines(file_content)
        f.close()

ants_dataset_train = MyData(
    "/home/cdk991014/workspace/Python/hymenoptera_data/train", "ants")
bees_dataset_train = MyData(
    "/home/cdk991014/workspace/Python/hymenoptera_data/train", "bees")

root_dir = "/home/cdk991014/workspace/Python/hymenoptera_data/train"
creat_label(root_dir, ants_dataset_train.img_path_list,
            ".txt", ants_dataset_train.label_dir)
creat_label(root_dir, bees_dataset_train.img_path_list,
            ".txt", bees_dataset_train.label_dir)
