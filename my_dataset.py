import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import glob
from PIL import Image
import random
import numpy as np
class Data_Loader(Dataset):
    #初始化读取图片
    def __init__(self, data_path, train):
        self.data_path = data_path
        self.train_img_path = glob.glob(os.path.join(data_path, 'train/image/*.png'))
        self.train_label_path = glob.glob(os.path.join(data_path, 'train/label/*.png'))
        self.test_img_path = glob.glob(os.path.join(data_path, 'test/*.png'))
        self.train = train
    #数据增强
    def change(self, image, rd):
        trans = []
        if self.train:
            if rd > 0.75:
                trans.append(transforms.RandomHorizontalFlip(1))
            elif rd > 0.5:
                trans.append(transforms.RandomVerticalFlip(1))
            elif rd > 0.25:
                trans.extend([
                    transforms.RandomHorizontalFlip(1),
                    transforms.RandomVerticalFlip(1),
                ])
        trans.extend([
            transforms.ToTensor(),
            # transforms.Normalize(mean=0.5, std=0.5)  # 通道数
        ])
        tr = transforms.Compose(trans)
        return tr(image)
    def __getitem__(self, index):
        if self.train:
            img_path = self.train_img_path[index]
            label_path = self.train_label_path[index]
            img = Image.open(img_path).convert('L')
            lbl = Image.open(label_path).convert('L')
            rd = random.random()
            img = self.change(img, rd)
            lbl = self.change(lbl, rd)
            return img, lbl
        else:
            label_path = self.test_img_path[index]
            lbl = Image.open(label_path).convert('L')
            lbl = self.change(lbl, 0)
            return lbl

    def __len__(self):
        if self.train:
            return len(self.train_img_path)
        return len(self.test_img_path)

# if __name__ == '__main__':
    # dataset = Data_Loader('..', True)
    # print("数据个数：", len(dataset))
#     print(dataset.__len__())
# #     # print(dataset.train_img_path[1])
# #     # img = Image.open(dataset.train_img_path[1]).convert('RGB')
# #     # print(img.size)
# #     #img = dataset.change(img, 0.8)
# #     #img.show()
# #     img, lbl= dataset.__getitem__(2)
# #     img = transforms.ToPILImage()(img)
# #     img.show()
#     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                batch_size=2,
#                                                shuffle=True)
#     for image, label in train_loader:
#         print(image.shape)