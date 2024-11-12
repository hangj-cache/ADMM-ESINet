from __future__ import print_function, division
import os
import torch
import argparse
from Network_Layers.ADMM_Network import ESINetADMMLayer
from utils.dataset import get_data_train
import torch.utils.data as data
import time
from datetime import datetime
from scipy.io import loadmat,savemat
# from os.path import join
import torch.nn as nn
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from torch.utils.tensorboard import SummaryWriter


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.abs(input - target).mean()

if __name__ == '__main__':

    ###############################################################################
    # parameters----argparse
    ###############################################################################
    parser = argparse.ArgumentParser(description=' main ')

    parser.add_argument('--data_dir', default='Data', type=str,
                        help='directory of data')
    parser.add_argument('--validation_data_dir',default='../data/training_set/data_deep_unfold')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='./logs_csnet', help='path of log files')
    parser.add_argument('--extents', type=str, default='1', help='signal noise ratio')
    parser.add_argument('--SNRs', type=str, default='5', help='signal noise ratio')
    parser.add_argument('--channels',default='62',help='choose the number of channel(Data volume)')
    parser.add_argument('--cond',type=str, default='various conditions', help='Conditions for selecting research')
    parser.add_argument('--V',type=str, default='V.mat',help='Variational operator')
    parser.add_argument('--result_dir',default='./result',type=str,help='the dir 0f reconstruct Source')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ###############################################################################

    def adjust_learning_rate(opt, epo, lr):
        """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
        lr = lr * (0.5 ** (epo // 10))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ###############################################################################
    # dataset
    ###############################################################################
    train, test, validate = get_data_train(args.data_dir,args.cond,args.SNRs)
    len_train, len_test, len_validate = len(train), len(test), len(validate)
    print("len_train: ", len_train, "\tlen_test:", len_test, "\tlen_validate:", len_validate)
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                   pin_memory=False)
    test_loader = data.DataLoader(dataset=test, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=False)
    valid_loader = data.DataLoader(dataset=validate, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=False)

    ###############################################################################

    L_path = ""
    L = torch.tensor(loadmat(L_path)['Gain']).to("cuda").float()

    #lambda1 = 10
    lambda = 1e4

    delta = 0.01  # 原来0.01
    ###############################################################################
    # ADMM-CSNET model
    ###############################################################################
    model = ESINetADMMLayer(L).cuda()


    num_params = count_parameters(model)
    print(f"Number of parameters:{num_params}")
    MAE = MAELoss()
    MSE = torch.nn.MSELoss(reduction='mean').cuda()

    ###############################################################################
    # Adam optimizer
    ###############################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ###############################################################################
    # self-define loss
    ###############################################################################
    # criterion = MyLoss().cuda()
    # writer = SummaryWriter(args.outf)
    ###############################################################################
    # train
    ###############################################################################
    # writer = SummaryWriter()
    print("start training...")
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 100000000

    for epoch in range(0, args.num_epoch + 1):
        print("EPOCH {}:".format(epoch+1))
        model.train(True)

        running_loss = 0.0
        last_loss = 0.0
        adjust_learning_rate(optimizer, epoch, lr=0.0001)

        for batch_idx, (s_real_trans, B_trans, Dic) in tqdm(enumerate(train_loader),desc='Training',unit='file'):
            ratio = 1

            s_real_trans = s_real_trans.to("cuda").float().squeeze(dim=0)
            B_trans = B_trans.to("cuda").float().squeeze(dim=0)
            Dic = Dic.to("cuda").float().squeeze(dim=0)

            optimizer.zero_grad()
            x = dict()
            x['B_trans'] = B_trans
            s_gen_trans = model(x)

            s_real = torch.matmul(s_real_trans, Dic)
            s_gen = torch.matmul(s_gen_trans, Dic)

            loss_S_mse = MSE(s_gen, s_real)
      
            loss = loss_S_mse

            running_loss += loss
            loss.backward()

            optimizer.step()

        last_loss = running_loss/len(train_loader)
        print("====================================================")
        print(f"mean_loss/epoch{last_loss}")

    ###############################################################################
    # validate
    ###############################################################################
        if epoch % 1 == 0:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx,(s_real_trans, B_trans, Dic) in tqdm(enumerate(valid_loader),desc='valid',unit='file'):
                    ratio = 1
                    # ratio = 1
                    s_real_trans = s_real_trans.to("cuda").float().squeeze(dim=0)
                    B_trans = B_trans.to("cuda").float().squeeze(dim=0)
                    Dic = Dic.to("cuda").float().squeeze(dim = 0)
                    # L = L.to("cuda").float().squeeze(dim=0)

                    x = dict()
                    x['B_trans'] = B_trans
                    s_gen_trans= model(x)
                    s_real = torch.matmul(s_real_trans, Dic)
                    s_gen = torch.matmul(s_gen_trans, Dic)

                    loss_S_mse = MSE(s_gen, s_real)
                
                    vloss = loss_S_mse

                    running_val_loss += vloss

            avg_val_loss = running_val_loss / len(valid_loader)
            best_vloss = last_loss
            print('m_LOSS train {} valid {}'.format(last_loss, avg_val_loss))
            model_path = 'ADMM_mean_model_{}_{}.pth'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.outf,args.cond,model_path))







