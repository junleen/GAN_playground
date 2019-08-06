import numpy as np
import struct
from tqdm import tqdm
import os

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    Args:
        idx3_ubyte_file: ubyte file path
    Return:
        np.ndarray
    """
    binary_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  #因为数据结构中前4行都是32位整数，采用i格式
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, binary_data, offset)
    print('number of images: %d, iamge size: %d*%d' % (num_images, num_rows, num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    
    fmt_image = '>' + str(image_size) + 'B'#图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    # print(fmt_image, offset, struct.calcsize(fmt_image))

    images = np.zeros((num_images, image_size), dtype=np.int32)
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, binary_data, offset=offset))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    Args:
        idx1_ubyte_file: ubyte_file path(labels)
    Return:
        np.ndarray, int32
    """
    binary_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, binary_data,offset)
    print('number of labels: %d' % (num_images))
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.zeros(num_images, dtype=np.int32)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, binary_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

class Mnist(object):
    def __init__(self, root):
        self.dataroot = root
        self._init_path()
        
    def _init_path(self):
        self.train_images_file = os.path.join(self.dataroot, 'train-images-idx3-ubyte')
        self.train_labels_file = os.path.join(self.dataroot, 'train-labels-idx1-ubyte')
        self.test_images_file = os.path.join(self.dataroot, 't10k-images-idx3-ubyte')
        self.test_labels_file = os.path.join(self.dataroot, 't10k-labels-idx1-ubyte')
    def read(self, is_train=True, one_hot=False):
        if is_train:
            images = decode_idx3_ubyte(self.train_images_file)
            labels = decode_idx1_ubyte(self.train_labels_file)
        else:
            images = decode_idx3_ubyte(self.test_images_file)
            labels = decode_idx1_ubyte(self.test_labels_file)
        if one_hot:
            labels_ = labels.copy()
            labels = np.zeros((len(labels), 10), dtype=np.int32)
            labels[np.arange(labels_.shape[0]), labels_] = 1
        return images, labels