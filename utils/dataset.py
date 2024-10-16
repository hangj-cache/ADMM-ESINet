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
        s_real_trans = torch.tensor(data['s_real_trans'])
        ActiveVoxSeed = data['ActiveVoxSeed'][0][0][0]
        ActiveVoxSeed_new = ActiveVoxSeed.astype(np.int16)
        B_trans = torch.tensor(data['B_trans'])
        Dic = torch.tensor(data['TBFs'])
        ratio = 1
        seedvox = data['seedvox'][0].item()
        # L = data['L']

        return s_real_trans / ratio, B_trans / ratio, seedvox, Dic, ActiveVoxSeed_new

    def __len__(self):
            return len(self.files)

def get_data_train(load_root, cond, SNRs):
    train = os.path.join(load_root, 'train', cond)
    test = os.path.join(load_root, 'test', cond)
    validate = os.path.join(load_root, 'validation', cond)
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data