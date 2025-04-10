# encoding:utf-8
import torch
import torch.utils.data as data
import os
import os.path
from scipy.io import loadmat
import numpy as np
class DataSet(data.Dataset):

    def __init__(self, dir):
        # files = os.listdir(dir)
        files = [file for file in os.listdir(dir) if 'data' in file]
        files.sort()
        self.files = [os.path.join(dir, file) for file in files]
        # self.num = [re.sub("\D", "", file) for file in files]   #\D表示不是数字的字符

    def __getitem__(self, index):
        file_path = self.files[index]
        data = loadmat(file_path)
        # s_real = torch.tensor(data['s_real_trans'])
        s_real = 1
        # ActiveVoxSeed = data['ActiveVoxSeed'][0][0][0]
        # ActiveVoxSeed_new = ActiveVoxSeed.astype(np.int16)
        B = torch.tensor(data['B'])
        Dic = torch.tensor(data['TBFs'])
        ratio = 1
        # seedvox = data['seedvox'][0].item()
        seedvox  = 1
        ActiveVoxSeed_new = 1

        return s_real / ratio, B / ratio, seedvox, Dic, ActiveVoxSeed_new

    def __len__(self):
            return len(self.files)

def get_data_train_position(load_root, cond, SNRs):
    train = os.path.join(load_root, 'train_1024', cond)
    test = os.path.join(load_root, 'test_1024', cond)
    validate = os.path.join(load_root, 'validation_1024', cond)
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data

def get_data_validation(load_root, cond, SNRs):
    train = os.path.join(load_root, cond, SNRs)
    test = os.path.join(load_root, cond, SNRs)
    validate = os.path.join(load_root,cond, SNRs)
    train_data = DataSet(train)
    test_data = DataSet(train)
    validate_data = DataSet(validate)
    return train_data,test_data,validate_data


def get_brainstrom_epilepsy_data(load_root, cond, SNRs):
    train = load_root
    test = load_root
    validate = load_root
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data


