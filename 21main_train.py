from __future__ import print_function, division
import os
import torch
import argparse
from Network_Layers.ADMM_Network import ESINetADMMLayer
# from utils.L21dataset import get_data_train
from utils.data_loader import get_data_train
import torch.utils.data as data
import time
from datetime import datetime
from scipy.io import loadmat,savemat
from os.path import join
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
    # parser.add_argument('--data_dir', default='./data/training_set/data_xin_4/', type=str,
    #                     help='directory of data')
    parser.add_argument('--data_dir', default='Data/DST_CedNet/', type=str,
                        help='directory of data')
    parser.add_argument('--validation_data_dir',default='../data/training_set/data_deep_unfold')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--localize_mi',default='Data/localize-mi_train_33times/subject01/')
    parser.add_argument('--simulation_data',default='Data/train_300_plus/')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs')
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
        lr = lr * (0.5 ** (epo // 25))  #original:50----每50个epoch调解一次学习率
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ###############################################################################
    # dataset
    ###############################################################################
    train_loader, valid_loader = get_data_train(args.localize_mi,args.cond,args.SNRs)

    ###############################################################################

    subject01_L_path = args.localize_mi + "validation\\various conditions\L.mat"
    L = torch.tensor(loadmat(subject01_L_path)['L']).to("cuda").float()


    lambda1 = 10   #original ：10
    lambda2 = 1e5  #original :150

    delta = 0.01  #原来0.01
    ###############################################################################
    # ADMM-CSNET model
    ###############################################################################
    model = ESINetADMMLayer(L).to("cuda")
    # model.reset_parameters()
    # # 统计需要更新的参数个数
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print("Total trainable parameters:", total_params)

    num_params = count_parameters(model)
    print(f"Number of parameters:{num_params}")
    MAE = MAELoss()
    MSE = torch.nn.MSELoss(reduction='mean').cuda()




    ###############################################################################
    # Adam optimizer
    ###############################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

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
        adjust_learning_rate(optimizer, epoch, lr=0.003)

        # ===================train==========================
        for batch_idx, (B_trans,s_real_trans,TBFs) in tqdm(enumerate(train_loader),desc='Training',unit='file'):
            ratio = 1

            s_real_trans = s_real_trans.to("cuda").float()
            B_trans = B_trans.to("cuda").float()
            TBFs = TBFs.to("cuda").float()

            TBFs_tp = torch.transpose(TBFs,1,2)

            optimizer.zero_grad()
            x = dict()
            x['B_trans'] = B_trans

            s_gen_trans = model(x)
            s_gen = torch.bmm(s_gen_trans,TBFs)
            s_real = torch.bmm(s_real_trans,TBFs)

            loss_S_mse = MSE(s_gen,s_real)
            loss = loss_S_mse * lambda2
            running_loss += loss

            # 进行反向传播
            loss.backward()


            optimizer.step()


        last_loss = running_loss/20000  #每个训练样本的平均损失
        print("====================================================")
        print(f"mean_loss/epoch{last_loss}")


    ###############################################################################
    # validate
    ###############################################################################
        if epoch % 1 == 0:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx,(B_trans,s_real_trans,TBFs) in tqdm(enumerate(valid_loader),desc='valid',unit='file'):
                    ratio = 1
                    # ratio = 1
                    s_real_trans = s_real_trans.to("cuda").float()
                    B_trans = B_trans.to("cuda").float()
                    TBFs = TBFs.to("cuda").float()

                    TBFs_tp = torch.transpose(TBFs,1,2)

                    x = dict()

                    x['B_trans'] = B_trans
                    s_gen_trans= model(x)  # 模型对象的输入是forward的输入
                    s_gen = torch.bmm(s_gen_trans, TBFs)
                    s_real= torch.bmm(s_real_trans,TBFs)

                    loss_S_mse = MSE(s_gen, s_real)

                    vloss = loss_S_mse * lambda2
                    running_val_loss += vloss


            avg_val_loss = running_val_loss / 200
            best_vloss = last_loss
            print('m_LOSS train {} valid {}'.format(last_loss, avg_val_loss))
            model_path = 'plus_1024-{}-6-2-0.003-sub01_model_{}_{}_1226.pth'.format(avg_val_loss,timestamp, epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.outf,args.cond,model_path))








