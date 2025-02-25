clc
clear
tic
%brainstorm %norgui
%%
%WGN = 1; % Using white Gaussian Nosie (WGN = 1) or Human EEG noise (WGN  = 0);
%LFNormlization = 0; % Whether normalize the LeadField Matrix
%Uniform = 1; % Uniform/NonUniform Sources

algorithms = {'VSSI-LpR', 'VSSI-L1', 'wMNE', 'LORETA', 'LPSSO', 'LASSO'};%, 'VSSI-L1', 'wMNE', 'LORETA','LPSSO'};%,VSSI-LpR 'VSSI-ARD'};% 'BESTIES'};%,'VSSI-CM','VSSI-MAP','Beamformer','Champagne','wMNE','LORETA'};%,'VSSI-CM','VSSI-MAP','Beamformer','Champagne','wMNE','LORETA'}%,'Champagne','MFOCUSS','sLORETA','dSPM'};%{'VSSI-CM','VSSI-Map','Champagne','MFOCUSS','sLORETA','dSPM'};%,'SBL','MFOCUSS','sloreta','dspm'}%,'VSSI-Mean','VSSI-MAP','SSI'}; 
% ResultsLoading = [0 0 0 0 0 0 1];
%定义了一个包含六个元素的向量，表示结果加载的情况。每个元素的值为1，表示结果将被加载。   
%ResultsLoading = [1 1 1 1 1 1];   %结果加载--这是对不同算法的结果加载   previously
ResultsLoading = [0 0 0 0 0 0];
WGN = 1; % Using white Gaussian Nosie (WGN = 1) or Human EEG noise (WGN  = 0);     %添加高斯白噪声
LFNormlization = 0; % Whether normalize the LeadField Matrix    %是否归一化L
Uniform = 1; % Uniform/NonUniform Sources   %   是否是一致的源
Test = 0;   %从这里开始即使一个个condition（只能选一个）      是否是测试
VariousExtents = 0;    %从这下面表示各种不同的情况   这格式不同尺寸
VariousSNRs = 0; cv_p = 0;   %不通过的信噪比----cv_p是用来判断交叉验证的  cross-validation：验证的就是不同的p值
VariousPatches = 0;   %这表示不同的脑区划分
VariousPvalues = 0;   %这表示
VariousCorrelation = 0;   %相关性
VariousChannels = 0;   %不同通道
VariousSNIRs = 0;

variousconditions = 1;

if variousconditions+VariousExtents+VariousSNRs+VariousPatches+VariousCorrelation+VariousPvalues+VariousChannels+Test ~= 1   %检查是否只是用了一个场景
    error('There will be one and only one scenario.');
end
% tic
%% Export the Channel,Cortex and HeadModel to workspace
if WGN
    channelselect=[1:32,34:42,44:64]; %Select EEG data
%     channelselect=[1:32 34:42 44:59 61:63];
else
    channelselect=[1:32 34:42 44:59 61:63]; % for Real noise simulations
end
% [sStudy, iStudy] = bst_get('StudyWithCondition', 'LiDaoli/Stim_2');
[sStudy, iStudy] = bst_get('StudyWithCondition','Subject01/Simulation');%GaussianSources');%Extents');%SNRs'); 
% [sStudy, iStudy] =bst_get('StudyWithCondition','Subject01/mind004_050924_median01_raw_clean_notch');
index = 1;
bst_call(@export_matlab, {char(sStudy.Data(1,index).FileName)},'data');
%=======================Import the LFM and Cortex==================================%
[sSurface, iSurface] = bst_get('SurfaceFileByType',[],'Cortex');
% [sSurface, iSurface] = bst_get('SurfaceFileByType',2,'Cortex');
bst_call(@export_matlab, {char(sSurface.FileName)},'Cortex');
Atlas = Cortex.Atlas(2);

[sHeadModel] = bst_get('HeadModelForStudy', iStudy);
bst_call(@export_matlab, {char(sHeadModel.FileName)},'model');
Gain=model.Gain(channelselect,:);
GridLoc=model.GridLoc;
GridOrient=model.GridOrient;

Gain = bst_gain_orient(Gain,GridOrient);
% load('MEGLeadfield.mat')
clear GridOrient
% load 'MEGLeadfield.mat';
% load 'CortexMEG.mat';
[nSensor,nSource] = size(Gain);
L = Gain;



if VariousSNRs
   scenario = 'various SNRs';%'various SNRs';
   SNR1 = [-5,0,5,10]; %原来[0,3,5,10]
   SNIR1 = zeros(4,1)+5;
   condition = SNR1';
   K = ones(5,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousSNIRs
   scenario = 'various conditions';
   SNR1 = zeros(5,1)+5;
   SNIR1 = [-5,0,5,10];%origin[0,3,5,10]
   condition = SNIR1';
   K = ones(5,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousExtents
   scenario = 'various conditions';
   SNR1 = 5*ones(5,1);
   SNIR1 = 5*ones(5,1);
   condition = [1:5]';
   K = ones(5,1);
   DefinedArea = [2 5 10 18 32]'*1e-4*ones(1,max(K));%[0.5 4 8 14 22 32]'*1e-4*ones(1,2);% 38 48]'*1e-4;
elseif VariousChannels
   scenario = 'various channels';
   SNR1 = 5*ones(4,1);
   SNIR1 = 5*ones(4,1);
   condition = [62, 46, 32, 16]';
   K = ones(4,1);
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif VariousPatches
   scenario = 'various patches';
   SNR1 = 5*ones(4,1);
   SNIR1 = 5*ones(4,1);
   condition = [1:4]';
   K = [1,2,3,4];
   DefinedArea = 8*1e-4*ones(size(condition,1),max(K));
elseif variousconditions
    scenario = 'various conditions';
    SNR1 = [-5,0,5,10];
    SNIR1 = [-5,0,5,10];
    condition = SNR1';
    K = [1,2,3,4,5];
    DefinedArea = [2 5 10 18 32]'*1e-4*ones(1,max(K));

elseif Test
    algorithms = {'SSSI-L2p'}%'VSSI-sL2p','VSSI-L2p'};%'VB-SCCD','SSSI-L2p','VSSI-GGD'
    ResultsLoading = [0 0 0];
    scenario = 'test';
    SNR1 = 0;
    SNIR1 = 0;
    condition = [1];
    K = 1;
    DefinedArea = 5*1e-4;
end

%% Output path
outpath = 'C:\Users\hangj\Desktop\L21_train_BATCH_SIZE\Data\train_300_plus\';  %每次生成数据路径记得改一下
% outpath = 'C:\Users\hangj\Desktop\L21_train_BATCH_SIZE\Data\two patch\8\';
set = 'train';
%if set == 'train'  %previous
if strcmp(set , 'train')  %要用strcmp进行比较，不要直接用'=='
    outpath = fullfile(outpath,set);
%elseif set == 'test'
elseif strcmp(set , 'test')
    outpath = fullfile(outpath,set);
else
    outpath = fullfile(outpath,'validation');
end

for i = 1 : size(condition,1)
%     path{i} = fullfile(outpath,scenario,'\',num2str(condition(i)));    %1x5  路径示例1：E:\result\VSSI-LpR\various SNRs\-10   path表示一个数组，包括了一种变量比如SNR的每一种condition的路径
    path{i} = fullfile(outpath,scenario);
    %fullfile(outpath, scenario, '\', num2str(condition(i))) 构建了一个完整的文件路径。这个路径由多个部分组成，它们通过 fullfile 函数合并在一起：
    if ~exist(path{i})
        mkdir(path{i});
    end
%      if ~exist([path{i} '\' 'metrics.mat'], 'file')
%          metrics = [];
%          save([path{i} '\' 'metrics.mat'], 'metrics')
%          % 将空的 metrics 变量保存为名为 "metrics.mat" 的 MAT 文件，该文件位于 path{i} 目录下。
%          % 这将创建一个新的 MAT 文件，并将其中包含一个名为 "metrics" 的变量，其值为空%
%      end
end

%outpath= 'F:\data_fake';
% outpath2 = 'F:\data1024\validation';
% outpath3 = 'F:\data1024\test';


%% Iteration

Eccentricity = sqrt(sum(GridLoc.^2,2));
Ec = find(Eccentricity > 70*1e-3);

for h = 1:2

for iteration = 1:size(condition,1)
    se = 1;
    Miter = 5000;

    B_totalSize = [5000, 62, 6];
    s_real_totalSize = [5000,1024,6];
    TBFs_totalSize = [5000,6,300];
    
    B_dataStorage = zeros(B_totalSize);
    s_real_dataStorage = zeros(s_real_totalSize);
    TBFs_dataStorage = zeros(TBFs_totalSize);
    for iter = 1:Miter
        ind = randperm(numel(Ec));

        ind_SNR = randi([1, 4]);
        ind_SNIR = randi([1, 4]);
        ind_extent = randi([1, 5]);

        SNR = SNR1(ind_SNR);
        SNIR = SNIR1(ind_SNIR);

        seedvox = Ec(ind(iteration)); 
%         seedvox = 971; 
        tau = [0.1 0.35 0.5 0.6];omega = [0.1 0.15 0.15 0.15];%[0.07 0.05 0.10 0.10];%[0.035 0.035 0.035 0.035];%*max(Time(Activetime));
    %tau = [0.1 0.2 0.5 0.6];omega = [0.1 0.15 0.15 0.15];%[0.07 0.05 0.10 0.10];%[0.035 0.035 0.035 0.035];%*max(Time(Activetime));
        f = [10 11 8 9];%10*ones(1,4);%5 (Hz);
        Amp = 1e-8;
        StimTime = find(abs(data.Time) == min(abs(data.Time)));
        TimeLen = 300;
%     Time = data.Time(StimTime-0.5*TimeLen+1:StimTime+0.5*TimeLen);
        %disp('StimTime:');
        %disp(StimTime);
        %disp('Length of data.Time:');
        %disp(length(data.Time));
        %Time = data.Time(StimTime-0.5*TimeLen + 1:StimTime+0.5*TimeLen);
        Time = data.Time(1:300);
    
        OPTIONS.DefinedArea    = DefinedArea(ind_extent,:);
        OPTIONS.seedvox        = seedvox;
        OPTIONS.frequency      = f;
        OPTIONS.tau            = tau;
        OPTIONS.omega          = omega;
        OPTIONS.Amp            = Amp;
        OPTIONS.GridLoc        = GridLoc;
        if VariousPatches  %混合矩阵MixMatrix是用来干嘛的？？？？
        OPTIONS.MixMatrix = [1    0    0     0;
                             0    1    0     0;
                             0    0    1     0;
                             0    0    0     1;
                             0    0    .5     .5];
        elseif VariousCorrelation
            xi = condition(iteration);
        %     OPTIONS.MixMatrix = [1-xi  xi   0    0;
        %                          0     1    0    0;
        %                          0  0   0    1;
        %                          0  0   1    0];
        OPTIONS.MixMatrix = [1            0            0          0;
                             xi     sqrt(1-xi^2)       0          0;
                             xi           0        sqrt(1-xi^2)   0;
                             0            0            1          0];
        end
        OPTIONS.uniform       = Uniform;
        OPTIONS.WGN           = WGN;

        OPTIONS.SNR           = SNR;
        OPTIONS.SNIR          = SNIR;

        OPTIONS.ar            = 0;
        OPTIONS.params(:,:,1) = [ 0.8    0    0 ;
                                    0  0.9  0.5 ;
                                  0.4    0  0.5];
    
        OPTIONS.params(:,:,2) = [-0.5    0    0 ;
                                    0 -0.8    0 ;
                                    0    0 -0.2];
    
        OPTIONS.noisecov      = [ 0.3    0    0 ;
                                    0    1    0 ;
                                    0    0  0.2];
        OPTIONS.SinglePoint   = 0;
    
        [B,s_real,Result] = Simulation_Data_Generate(L,Cortex,Time,OPTIONS);
        %s_real= awgn(s_real,20,'measured');
     % s_real= awgn(s_real,20,'measured'); %背景噪声  成像后，再用这个来计算AUC
     
      %s_real(Result.ActiveVox,:)=s_real(Result.ActiveVox,:)+8e-10; 
%      source幅值
        ratio = 1;
%         ratio = norm(B,'fro');
%         result_str = sprintf('%.1e', ratio);
%         ratio = str2double(result_str);

        ActiveVoxSeed = Result.ActiveVoxSeed;
        seedvox = Result.seedvox;
        
        
    % %=======================================================================%
        fprintf('Actual SNR is %g\n',20*log10(norm(L*s_real,'fro')/norm(B-L*s_real,'fro')));
        %if se <= 18500
        %data_name = sprintf('data_%d.mat', se);
        %sreal_name = sprintf('sreal_%d.mat', se);
        %save([path{iteration} '\' data_name], 'B', 's_real')
%         datayu_name = sprintf('datayu_%d.mat', se)

        %idx = max(s_real);
        %[~,pos] = max(idx);
        %用MNE对脑电信号进行预处理
        %% 对原始的脑电信号进行放大处理
%         ratio = 1e3;
        B_amp = B./ratio;
        s_real_amp = s_real./ratio;
        %% for original data
        [junk,v,d] = svd(B,'econ');
        Eigvals = diag(v).^2;
        NEigvals = Eigvals./sum(Eigvals);
        %% TBFselect
        KSVD = 6;%sum(NEigvals' >= threshold);
        TBFs = d(:,1:KSVD)';

        B_trans = B_amp*TBFs';
        s_real_trans = s_real_amp*TBFs';

        %[Nsensor,Nsource] = size(L);
%         T = size(B_trans,2);                  %T sample points
        V = VariationEdge(Cortex.VertConn);

        %s_real = s_real;
        %s_real_trans = s_real_amp * V_d;

%         TMNE = MNE(B_amp,[],Gain,[],'MNE'); 
%         s_MNE = TMNE*B_amp; %这是预处理得到的源S                                                           
%         s_yu_trans = s_MNE*TBFs';   %为这个源添加噪声   这是一个列向量
        %s_yu_trans = s_yu;
%         save([path{iteration} '\' datayu_name], 's_yu_amp','s_real_amp',"B_amp","ActiveVoxSeed","seedvox");
        
        B_dataStorage(se,:,:) = B_trans;
        s_real_dataStorage(se,:,:) = s_real_trans;
        TBFs_dataStorage(se,:,:) = TBFs;
        

%         save([path{iteration} '\' datayu_name],"B_trans","ActiveVoxSeed","seedvox","TBFs");    

        %save([path{iteration} '\V.mat'], "V");


%    data_name = sprintf('data_%d.mat', se);
%    sreal_name = sprintf('sreal_%d.mat', se);
%     
%    save([outpath '\' data_name], 'B', 's_real')
%     elseif se <= 19250
%         data_name = sprintf('data_%d.mat', se);
%         sreal_name = sprintf('sreal_%d.mat', se);
%     
%         save([outpath2 '\' data_name], 'B', 's_real')
%     else
%          data_name = sprintf('data_%d.mat', se);
%         sreal_name = sprintf('sreal_%d.mat', se);
%     
%         save([outpath3 '\' data_name], 'B', 's_real')
        folderpath = path{iteration};
        filenames = ["GridLoc.mat";"Cortex.mat";"L.mat";"model.mat"];
        filenames2 = ["GridLoc";"Cortex";"L";"model"];
        for a = 1:length(filenames)
            fullpath = fullfile(path(iteration),filenames(a))
            if exist('fullpath', 'file') == 2
                continue;
            else
                save(char(fullpath),char(filenames2(a)))
                %save([char(fullpath)], filename2(a));
            end
        end
        se = se + 1;
    end
%     se = se + 1;
% end
save([path{iteration} '\' 'datayu_' num2str(4*(h-1)+iteration) '.mat'],'B_dataStorage','s_real_dataStorage','TBFs_dataStorage');
end
end

% save([outpath1 '\' 'GridLoc.mat'], 'GridLoc')
% save([outpath1 '\' 'Cortex.mat'], 'Cortex')
% save([outpath1 '\' 'L.mat'], 'Gain')
% save([outpath2 '\' 'GridLoc.mat'], 'GridLoc')
% save([outpath2 '\' 'Cortex.mat'], 'Cortex')
% save([outpath2 '\' 'L.mat'], 'Gain')    
% save([outpath3 '\' 'GridLoc.mat'], 'GridLoc')
% save([outpath3 '\' 'Cortex.mat'], 'Cortex')
% save([outpath3 '\' 'L.mat'], 'Gain')    
    
    
    