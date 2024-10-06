#!/usr/bin/env python
# coding: utf-8

# In[5]:


from osgeo import gdal
import numpy as np
import os
# os.environ['PROJ_LIB'] = r'C:\Users\Lenovo\.conda\envs\zph\Library\share\proj'
# os.environ['GDAL_DATA'] = r'C:\Users\Lenovo\.conda\envs\zph\Library\share'
# gdal.PushErrorHandler("CPLQuietErrorHandler")


class ImageProcess:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(self.filepath, gdal.GA_ReadOnly)
        self.info = []
        self.img_data = None
        self.data_8bit = None

    def read_img_info(self):
        # 获取波段、宽、高
        img_bands = self.dataset.RasterCount
        img_width = self.dataset.RasterXSize
        img_height = self.dataset.RasterYSize
        # 获取仿射矩阵、投影
        img_geotrans = self.dataset.GetGeoTransform()
        img_proj = self.dataset.GetProjection()
        self.info = [img_bands, img_width, img_height, img_geotrans, img_proj]
        return self.info

    def read_img_data(self):
        self.img_data = self.dataset.ReadAsArray(0, 0, self.info[1], self.info[2])
        return self.img_data

    # 影像写入文件
    @staticmethod
    def write_img(filename: str, img_data: np.array, **kwargs):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(img_data.shape) >= 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
        # 写入仿射变换参数
        if 'img_geotrans' in kwargs:
            outdataset.SetGeoTransform(kwargs['img_geotrans'])
        # 写入投影
        if 'img_proj' in kwargs:
            outdataset.SetProjection(kwargs['img_proj'])
        # 写入文件
        if img_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(img_data)  # 写入数组数据
        else:
            for i in range(img_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del outdataset


# In[6]:


def read_multi_bands(image_path):
    """
    读取多波段文件
    :param image_path: 多波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    image = ImageProcess(filepath=image_path)
    # 读取影像元信息
    image_info = image.read_img_info()
    print(f"多波段影像元信息：{image_info}")
    # 读取影像矩阵
    image_data = image.read_img_data()
    print(f"多波段矩阵大小：{image_data.shape}")
    return image, image_info, image_data




# In[7]:


# 单波段的images
def read_single_band(band_path):
    """
    读取单波段文件
    :param band_path: 单波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    band = ImageProcess(filepath=band_path)
    # 读取影像元信息
    band_info = band.read_img_info()
    print(f"单波段影像元信息：{band_info}")
    # 读取影像矩阵
    band_data = band.read_img_data()
    print(f"单波段矩阵大小：{band_data.shape}")
    return band, band_info, band_data


import math
import numpy as np
from alive_progress import alive_bar
# from module.image import *


def cal_single_band_slice(single_band_data, slice_size=512):
    """
    计算单波段的格网裁剪四角点
    :param single_band_data:单波段原始数据
    :param slice_size: 裁剪大小
    :return: 嵌套列表，每一个块的四角行列号
    """
    single_band_size = single_band_data.shape
    row_num = math.ceil(single_band_size[0] / slice_size)  # 向上取整
    col_num = math.ceil(single_band_size[1] / slice_size)  # 向上取整
    print(f"行列数：{single_band_size}，行分割数量：{row_num}，列分割数量：{col_num}")
    slice_index = []
    for i in range(row_num):
        for j in range(col_num):
            row_min = i * slice_size
            row_max = (i + 1) * slice_size
            if (i + 1) * slice_size > single_band_size[0]:
                row_max = single_band_size[0]
            col_min = j * slice_size
            col_max = (j + 1) * slice_size
            if (j + 1) * slice_size > single_band_size[1]:
                col_max = single_band_size[1]
            slice_index.append([row_min, row_max, col_min, col_max])
    return slice_index





def single_band_slice(single_band_data, index=[0, 1000, 0, 1000], slice_size=1000, edge_fill=False):
    """
    依据四角坐标，切分单波段影像
    :param single_band_data:原始矩阵数据
    :param index: 四角坐标
    :param slice_size: 分块大小
    :param edge_fill: 是否进行边缘填充
    :return: 切分好的单波段矩阵
    """
    if edge_fill:
        if (index[1] - index[0] != slice_size) or (index[3] - index[2] != slice_size):
            result = np.empty(shape=(slice_size, slice_size))
            new_row_min = index[0] % slice_size
            new_row_max = new_row_min + (index[1] - index[0])
            new_col_min = index[2] % slice_size
            new_col_max = new_col_min + (index[3] - index[2])
            result[new_row_min:new_row_max, new_col_min:new_col_max] = single_band_data[index[0]:index[1],
                                                                       index[2]:index[3]]
        else:
            result = single_band_data[index[0]:index[1], index[2]:index[3]]
    else:
        result = single_band_data[index[0]:index[1], index[2]:index[3]]
    return result.astype(single_band_data.dtype)



def slice_conbine(slice_all, slice_index):
    """
    将分块矩阵进行合并
    :param slice_all: 所有的分块矩阵列表
    :param slice_index: 分块的四角坐标
    :return: 合并的矩阵
    """
    combine_data = np.zeros(shape=(slice_index[-1][1], slice_index[-1][3]))
    # print(combine_data.shape)
    for i, slice_element in enumerate(slice_index):
        combine_data[slice_element[0]:slice_element[1], slice_element[2]:slice_element[3]] = slice_all[i]
    return combine_data


def coordtransf(Xpixel, Ypixel, GeoTransform):
    """
    像素坐标和地理坐标仿射变换
    :param Xpixel: 左上角行号
    :param Ypixel: 左上角列号
    :param GeoTransform: 原始仿射矩阵
    :return: 新的仿射矩阵
    """
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    slice_geotrans = (XGeo, GeoTransform[1], GeoTransform[2], YGeo, GeoTransform[4], GeoTransform[5])
    return slice_geotrans


def single_band_grid_slice(band_path, band_slice_dir, slice_size, edge_fill=False):
    """
    单波段格网裁剪
    :param band_path: 原始单波段影像
    :param band_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :return:
    """
    band, band_info, band_data = read_single_band(band_path)
    # 计算分块的四角行列号
    slice_index = cal_single_band_slice(band_data, slice_size=slice_size)
    # 执行裁剪
    with alive_bar(len(slice_index), force_tty=True) as bar:
        for i, slice_element in enumerate(slice_index):
            slice_data = single_band_slice(band_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪单波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], band_info[3])  # 转换仿射坐标
            band.write_img(band_slice_dir + r'\single_grid_slice_' + str(i) + '.tif', slice_data,
                           img_geotrans=slice_geotrans, img_proj=band_info[4])  # 写入文件
            bar()
        print('单波段格网裁剪完成')


# In[8]:


# 多波段的images
def read_multi_bands(image_path):
    """
    读取多波段文件
    :param image_path: 多波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    image = ImageProcess(filepath=image_path)
    # 读取影像元信息
    image_info = image.read_img_info()
    print(f"多波段影像元信息：{image_info}")
    # 读取影像矩阵
    image_data = image.read_img_data()
    print(f"多波段矩阵大小：{image_data.shape}")
    return image, image_info, image_data


import math
import numpy as np
from alive_progress import alive_bar

def cal_single_band_slice(single_band_data, slice_size=512):
    """
    计算单波段的格网裁剪四角点
    :param single_band_data:单波段原始数据
    :param slice_size: 裁剪大小
    :return: 嵌套列表，每一个块的四角行列号
    """
    single_band_size = single_band_data.shape
    row_num = math.ceil(single_band_size[0] / slice_size)  # 向上取整
    col_num = math.ceil(single_band_size[1] / slice_size)  # 向上取整
    print(f"行列数：{single_band_size}，行分割数量：{row_num}，列分割数量：{col_num}")
    slice_index = []
    for i in range(row_num):
        for j in range(col_num):
            row_min = i * slice_size
            row_max = (i + 1) * slice_size
            if (i + 1) * slice_size > single_band_size[0]:
                row_max = single_band_size[0]
            col_min = j * slice_size
            col_max = (j + 1) * slice_size
            if (j + 1) * slice_size > single_band_size[1]:
                col_max = single_band_size[1]
            slice_index.append([row_min, row_max, col_min, col_max])
    return slice_index





def multi_bands_slice(multi_bands_data, index=[0, 512, 0, 512], slice_size=512, edge_fill=False):
    """
    依据四角坐标，切分多波段影像
    :param multi_bands_data: 原始多波段矩阵
    :param index: 四角坐标
    :param slice_size: 分块大小
    :param edge_fill: 是否进行边缘填充
    :return: 切分好的多波段矩阵
    """
    if edge_fill==True:
        if (index[1] - index[0] != slice_size) or (index[3] - index[2] != slice_size):
            result = np.empty(shape=(multi_bands_data.shape[0], slice_size, slice_size))
            new_row_min = index[0] % slice_size    # 0
            new_row_max = new_row_min + (index[1] - index[0])  
            new_col_min = index[2] % slice_size    # 0
            new_col_max = new_col_min + (index[3] - index[2])
            result[:, new_row_min:new_row_max, new_col_min:new_col_max] = multi_bands_data[:, index[0]:index[1],
                                                                          index[2]:index[3]]
        else:
            result = multi_bands_data[:, index[0]:index[1], index[2]:index[3]] 
    else:
        result = multi_bands_data[:, index[0]:index[1], index[2]:index[3]]
    return result.astype(multi_bands_data.dtype)


def slice_conbine(slice_all, slice_index):
    """
    将分块矩阵进行合并
    :param slice_all: 所有的分块矩阵列表
    :param slice_index: 分块的四角坐标
    :return: 合并的矩阵
    """
    combine_data = np.zeros(shape=(slice_index[-1][1], slice_index[-1][3]))
    # print(combine_data.shape)
    for i, slice_element in enumerate(slice_index):
        combine_data[slice_element[0]:slice_element[1], slice_element[2]:slice_element[3]] = slice_all[i]
    return combine_data


def coordtransf(Xpixel, Ypixel, GeoTransform):
    """
    像素坐标和地理坐标仿射变换
    :param Xpixel: 左上角行号
    :param Ypixel: 左上角列号
    :param GeoTransform: 原始仿射矩阵
    :return: 新的仿射矩阵
    """
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    slice_geotrans = (XGeo, GeoTransform[1], GeoTransform[2], YGeo, GeoTransform[4], GeoTransform[5])
    return slice_geotrans


def multi_bands_grid_slice(image_path, image_slice_dir, slice_size, edge_fill=False):
    """
    多波段格网裁剪
    :param image_path: 原始多波段影像
    :param image_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :return:
    """
    image, image_info, image_data = read_multi_bands(image_path)

    # 计算分块的四角行列号
    slice_index = cal_single_band_slice(image_data[0, :, :], slice_size=slice_size)
    # 执行裁剪
    with alive_bar(len(slice_index), force_tty=True) as bar:
        for i, slice_element in enumerate(slice_index):
            slice_data = multi_bands_slice(image_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪多波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], image_info[3])  # 转换仿射坐标
            image.write_img(image_slice_dir + r'\multi_grid_slice_' + str(i) + '.tif', slice_data,
                            img_geotrans=slice_geotrans, img_proj=image_info[4])  # 写入文件
            bar()
        print('多波段格网裁剪完成')


# In[9]:


# 创建好带有trainning和label的文件夹
root = "E:\\unet_complete_version\\raw_material"

folder1 = os.path.join(root, 'dataset')
if not os.path.exists(folder1): os.mkdir(folder1)
foldertraining = os.path.join(folder1, 'trainning')
if not os.path.exists(foldertraining): os.mkdir(foldertraining)

train_img_folder = os.path.join(foldertraining, 'imgs')
if not os.path.exists(train_img_folder): os.mkdir(train_img_folder)

train_label_folder = os.path.join(foldertraining, 'labels')
if not os.path.exists(train_label_folder): os.mkdir(train_label_folder)


# In[10]:


# 调用上面的multi_bands_grid_slice和single_band_grid_slice函数进行剪切
image_path = "E:\\unet_complete_version\\raw_material\\naip.tif"
image_slice_dir = train_img_folder   
slice_size = 512
edge_fill = True
slice_train = multi_bands_grid_slice(image_path, image_slice_dir, slice_size, edge_fill=edge_fill)

label_path = "raw_material\\landuse.tif"
image_slice_dir = train_label_folder
slice_size = 512
edge_fill = True
slice_label= single_band_grid_slice(label_path, image_slice_dir, slice_size, edge_fill=edge_fill)    #一般单波段的影像是用来当label


# In[11]:


import os 
valid_folder = os.path.join(folder1, 'validation')
if not os.path.exists(valid_folder): os.mkdir(valid_folder)
valid_label_folder = os.path.join(valid_folder, 'labels')
if not os.path.exists(valid_label_folder): os.mkdir(valid_label_folder)
valid_img_folder = os.path.join(valid_folder, 'imgs')
if not os.path.exists(valid_img_folder): os.mkdir(valid_img_folder)


# In[12]:


# 划分验证集和训练集
import random
import shutil
import os
from path import Path


def split_train_validate(training_img_dir, training_labels_dir, validation_img_dir, validation_labels_dir, split = 10):
    '''split the training dataset into training and validationg parts, randomly
    select 10% of the training datast (imgs and labels) into the validation folder
    '''
    ## split the train dataset and validation dataset
    img_sample = random.sample(Path(training_img_dir).files(), len(Path(training_img_dir).files())//split )
    label_sample = random.sample(Path(training_labels_dir).files(), len(Path(training_labels_dir).files())//split )

    if not os.path.exists(validation_img_dir): os.mkdir(validation_img_dir)
    if not os.path.exists(validation_labels_dir): os.mkdir(validation_labels_dir)

    for i,j in zip(img_sample,label_sample):
        shutil.move(os.path.join(training_img_dir, i.name), os.path.join(validation_img_dir, i.name))
        shutil.move(os.path.join(training_labels_dir, j.name), os.path.join(validation_labels_dir, j.name))
 
# 创建验证集的文件夹
valid_folder = os.path.join(folder1, 'validation')
if not os.path.exists(valid_folder): os.mkdir(valid_folder)
valid_label_folder = os.path.join(valid_folder, 'labels')
if not os.path.exists(valid_label_folder): os.mkdir(valid_label_folder)
valid_img_folder = os.path.join(valid_folder, 'imgs')
if not os.path.exists(valid_img_folder): os.mkdir(valid_img_folder)


split_train_validate(train_img_folder, train_label_folder, valid_img_folder, valid_label_folder)


# In[13]:


import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import torch
from pathlib import Path
import matplotlib.pyplot as plt


# this is used to augment the image
transform_aug = transforms.Compose([
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5)
])

class CanopyDataset(Dataset):
    def __init__(self, img_dir, msk_dir, pytorch=True, transforms=None):
        super().__init__()
        
        img_files = [f for f in img_dir.iterdir() if not f.is_dir()]
        mask_files = [f for f in msk_dir.iterdir() if not f.is_dir()]

        self.files = [self.combine_files(img, mask) for img, mask in zip(img_files, mask_files)]
        self.pytorch = pytorch
        self.transforms = transforms

    def combine_files(self, img_file: Path, mask_file: Path):
        files = {'image': img_file, 'mask': mask_file}
        return files
    
    def __len__(self):
        return len(self.files)
    
    def open_as_array(self, idx):
        image_path = str(self.files[idx]['image'])
        image = cv2.imread(image_path)

        
        image = cv2.resize(image, (512, 512))
        image = image.transpose((2, 0, 1)) / 255.0
        
        return image
    
    def open_mask(self, idx, add_dims=False):
        mask_path = str(self.files[idx]['mask'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.imread(mask_path)
        if image is None:
            raise ValueError(f"Image at {mask_path} could not be loaded.")
        mask = cv2.resize(mask, (512, 512))
        mask = np.where(mask == 5, 1, 0)
        
        if add_dims:
            mask = np.expand_dims(mask, 0)
        
        return mask

    def __getitem__(self, idx):
        image = self.open_as_array(idx)
        mask = self.open_mask(idx, add_dims=False)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.int64)
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__ ())
        return s
    


base_path = Path('raw_material\\dataset\\trainning')
data = CanopyDataset(base_path/'imgs', base_path/'labels')

def check(data):
    sample = data.__getitem__(0)
    img = sample[0]
    label = sample[1]
    sample = label.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.numpy()
    # 创建一个包含两个子图的图形
    fig, axes = plt.subplots(1, 2)

    # 在第一个子图中显示第一幅图像
    axes[0].imshow(label)
    axes[0].set_title('label')

    # 在第二个子图中显示第二幅图像
    axes[1].imshow(img)
    axes[1].set_title('img')

    # 调整布局，避免图像重叠
    plt.tight_layout()

    # 显示图形
    plt.show()



try:
    check(data)
except ValueError as e:
    print(e)


train_ds, valid_ds,test_ds = random_split(data, (70,10,10))
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=True)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=True)
print('train_dl is:', len(train_dl))
print(type(train_dl))
print('valid_dl is:', len(valid_dl))
print('test_dl is:', len(test_dl))


# In[14]:


import torch.nn as nn
import torch
import numpy as np


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #parameters: in_channels, out_channels, kernel_size, padding
        self.conv1 = self.encoder(in_channels, 64, 3, 1)
        self.conv2 = self.encoder(64, 128, 3, 1)
        self.conv3 = self.encoder(128, 256, 3, 1)
        
        self.upconv3 = self.decoder(256, 128, 3, 1)
        self.upconv2 = self.decoder(128*2, 64, 3, 1)
        self.upconv1 = self.decoder(64*2, out_channels, 3, 1)
        
    # will be call when create instance
    def __call__(self, x): 
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        return upconv1
    

    # ---------------------------------------------------------------------------
    
    def encoder(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding= padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        
        return contract
    
    def decoder(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand


# In[15]:


import time
def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs):
    start = time.time()
    model.cuda()
    
    train_loss, valid_loss = [], []
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        
        # print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl


            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()


                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                        
                # stats - whatever is the phase
                acc = acc_fn(outputs, y)
                
                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            
            # clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)
            

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
    

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


unet = UNET(3,8)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(), lr=0.01)
train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=2)

tem = torch.save(unet.state_dict(), 'unet_build_model100epc_aug.pth')


# In[21]:


model = UNET(3,8)

model_path = r'unet_build_model100epc_aug.pth'
model.load_state_dict(torch.load(model_path))

model.eval()
print(len(test_dl))


total_acc = 0  
num_batches = 0  
with torch.no_grad():
    for xb,yb in test_dl:
        output = model(xb).to('cuda:0') 
        acc = acc_metric(output,yb)
        total_acc += acc
        num_batches+=1

average_acc = total_acc / num_batches  
print(f"Average Accuracy: {average_acc}")
        
# print(output.shape)
# ## batch size
# bs = 1
# fig, ax = plt.subplots(bs, 3, figsize=(15, bs*5))
# for i in range(bs):
#     ax[i,0].imshow(batch_to_img(xb,i))
#     ax[i,1].imshow(yb[i])
#     ax[i,2].imshow(predb_to_mask(output, i))
    
if __name__ == "__main__":
    UNET()
