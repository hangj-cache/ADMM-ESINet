from __future__ import print_function, division
import os
import torch
import argparse
from Network_Layers.ADMM_Network import ESINetADMMLayer
from utils.L21dataset import get_data_validation
import torch.utils.data as data
from scipy.io import loadmat,savemat
from os.path import join
import torch.nn as nn
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from torch.utils.tensorboard import SummaryWriter
import time


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.abs(input - target).mean()

if __name__ == '__main__':

    ###############################################################################
    # parameters----argparse一般只要一个放在main中--要用的全部参数
    ###############################################################################
    parser = argparse.ArgumentParser(description=' main ')
    # parser.add_argument('--data_dir', default='./data/training_set/data_xin_4/', type=str,
    #                     help='directory of data')
    parser.add_argument('--data_dir', default='Data', type=str,
                        help='directory of data')
    # parser.add_argument('--validation_data_dir',default='Data\\localize-mi_trans_33times\\subject01\\')
    parser.add_argument('--validation_data_dir',default='Data\审稿数据\\5006_validation\\')
    # parser.add_argument('--validation_data_dir',default='Data\审稿数据\\1024_validation\\')
    parser.add_argument('--brainstorm_epilepsy',default='Data\\brainstorm_ecilepsy_data')
    parser.add_argument('--yokagawa',default='Data\\Yokagawa\\real_data\\')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='./logs_csnet', help='path of log files')
    parser.add_argument('--SNRs', type=str, default='-10', help='signal noise ratio')
    parser.add_argument('--extents', type=str, default='1', help='area of the source')
    parser.add_argument('--patchs', type=str, default='1', help='area of the source')
    parser.add_argument('--channels',default='62',help='choose the number of channel(Data volume)')
    parser.add_argument('--cond',type=str, default='various Extents', help='Conditions for selecting research')
    parser.add_argument('--V',type=str, default='V.mat',help='Variational operator')
    parser.add_argument('--result_dir',default='./result\\审稿意见\\',type=str,help='the dir 0f reconstruct Source')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ###############################################################################

    def adjust_learning_rate(opt, epo, lr):
        """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
        lr = lr * (0.5 ** (epo // 25))  #original:50----每50个epoch调解一次学习率
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ###############################################################################
    # dataset
    ###############################################################################
    train, test, validate = get_data_validation(args.validation_data_dir,args.cond,args.extents)
    len_train, len_test, len_validate = len(train), len(test), len(validate)
    print("len_train: ", len_train, "\tlen_test:", len_test, "\tlen_validate:", len_validate)
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)   #上面num_workers说是建议在windows上设为0
    test_loader = data.DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  pin_memory=False)
    valid_loader = data.DataLoader(dataset=validate, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)

    ###############################################################################
    # mask
    ###############################################################################
    dir = join(args.validation_data_dir,args.cond)

    data = loadmat(join(dir ,args.extents , 'L.mat'))
    mask = data['L']   #L


    mask = torch.tensor(mask, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    mask = mask.float()


    lambda1 = 10   #original ：10
    lambda2 = 1e4  #original :150
    delta = 0.1  #原来0.01
    ###############################################################################
    # ADMM-CSNET model
    ###############################################################################
    model = ESINetADMMLayer(mask).cuda()
    # model.reset_parameters()
    # # 统计需要更新的参数个数
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print("Total trainable parameters:", total_params)

    num_params = count_parameters(model)
    print(f"Number of parameters:{num_params}")
    MAE = MAELoss()
    MSE = torch.nn.MSELoss(reduction='mean').cuda()


    model_params = join("logs_csnet","various conditions","yokogawa-6d-2d-256-600-0.001-_model_0.15674690902233124_20250325_095337.pth")
    if os.path.exists(model_params):
        params_load = torch.load(model_params)
        model.load_state_dict(params_load)

        print("==================validation======================")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (s_real_trans, B_trans, seedvox, TBFs, ActiveVoxSeed) in tqdm(enumerate(valid_loader), desc='valid',
                                                                          unit='file'):
                ratio = 1
                # ratio = 1
                s_real_trans = s_real_trans.to("cuda").float().squeeze(dim=0)
                B_trans = B_trans.to("cuda").float().squeeze(dim=0)
                TBFs = TBFs.to("cuda").float().squeeze(dim=0)
                seedvox = seedvox.to("cuda").squeeze(dim=0)
                x = dict()
                x['B_trans'] = B_trans.unsqueeze(0)
                # x['L'] = mask
                # x['zuobiao'] = zuobiao_matrix
                # x['V'] = V
                # x['s_yu_amp'] = s_yu_amp
                start_time = time.time()
                s_gen_trans = model(x)  # 模型对象的输入是forward的输入
                end_time = time.time()
                print(end_time - start_time)
                # V_dt = V_d.t()
                s_gen = torch.matmul(s_gen_trans, TBFs)
                s_real = torch.matmul(s_real_trans, TBFs)


                filename = os.path.join(args.result_dir,args.cond,args.extents, f'result_{batch_idx}.mat')

                s_gen = s_gen * ratio
                s_real = s_real * ratio
                savemat(filename,
                        {'reco_S': s_gen.cpu().numpy(), 'real_S': s_real.cpu().numpy(),
                         'B_trans': B_trans.cpu().numpy(), 's_real_trans': s_real_trans.cpu().numpy(),
                         'seedvox': seedvox.cpu().numpy(), 'TBFs': TBFs.cpu().numpy(),'ActiveVoxSeed':ActiveVoxSeed.cpu().numpy()})




