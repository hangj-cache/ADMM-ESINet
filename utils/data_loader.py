
# encoding:utf-8
import torch
import torch.utils.data as data
import os
import os.path
from scipy.io import loadmat
import numpy as np
class DataSet(data.Dataset):

    def __init__(self, B_trans,s_real_trans,TBFs):
        self.B_trans = B_trans
        self.s_real_trans = s_real_trans
        self.TBFs = TBFs

    def __getitem__(self, index):

        return self.B_trans[index],self.s_real_trans[index],self.TBFs[index]

    def __len__(self):
        return self.B_trans.shape[0]

def get_data_train(load_root, cond, SNRs):
    train = os.path.join(load_root, 'train', cond)
    validate = os.path.join(load_root, 'validation', cond)
    for i in range(4):
        tr_data = loadmat(train + '/datayu_xin_' + str(i + 1) + '.mat')
        val_data = loadmat(validate + '/datayu_xin_' + str(i + 1) + '.mat')
        if i == 0:
            x = tr_data['B_dataStorage']
            y = tr_data['s_real_dataStorage']
            z = tr_data['TBFs_dataStorage']
            B_tr = x
            s_real_tr = y
            TBFs_tr = z

            m = val_data['B_dataStorage']
            n = val_data['s_real_dataStorage']
            l = val_data['TBFs_dataStorage']
            B_val = m
            s_real_val = n
            TBFs_val = l

        else:
            x = tr_data['B_dataStorage']
            y = tr_data['s_real_dataStorage']
            z = tr_data['TBFs_dataStorage']
            B_tr = np.concatenate((B_tr, x))  # 3-D tensor 的拼接
            s_real_tr = np.concatenate((s_real_tr, y))
            TBFs_tr = np.concatenate((TBFs_tr,z))

            m = val_data['B_dataStorage']
            n = val_data['s_real_dataStorage']
            l = val_data['TBFs_dataStorage']
            B_val = np.concatenate((B_val, m))
            s_real_val = np.concatenate((s_real_val, n))
            TBFs_val = np.concatenate((TBFs_val,l))

    train_data = DataSet(B_tr,s_real_tr,TBFs_tr)
    validate_data = DataSet(B_val,s_real_val,TBFs_val)
    train_loader = data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=0,
                                   pin_memory=False)  # 上面num_workers说是建议在windows上设为0
    valid_loader = data.DataLoader(dataset=validate_data, batch_size=16, shuffle=False, num_workers=0,
                                   pin_memory=False)
    return train_loader, valid_loader

def get_data_validation(load_root, cond, SNRs):
    train = os.path.join(load_root, cond, SNRs)
    test = os.path.join(load_root,cond, SNRs)
    validate = os.path.join(load_root,cond, SNRs)
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data


def get_brainstrom_epilepsy_data(load_root, cond, SNRs):
    train = load_root
    test = load_root
    validate = load_root
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data



def get_localizemi_data(load_root, cond, SNRs):
    train = load_root
    test = load_root
    validate = load_root
    train_data = DataSet(train)
    validate_data = DataSet(validate)
    return train_data, validate_data


