function [L,B,s_real]=Simulate_data(Cortex,Gain,SNR,ActiveVoxSeed,seedvox,SNIR)

[nSensor,nSource] = size(Gain);
%% Generate Simulated EEG Data
%------------------ Simulation data ---------------------%

[~, VertArea] = tess_area(Cortex.Vertices, Cortex.Faces);
Time = -0.996:1/250:1;
StimTime = find(abs(Time) == min(abs(Time)));

s_real = zeros(nSource,numel(Time));
f = 5;
Activetime = StimTime:numel(Time);
% -----------Gaussian Damped sinusoidal time courses------------------%
tao = [0.20 0.22 0.25 0.28]*max(Time(Activetime));
omega = [0.06 0.06 0.06 0.06]*max(Time(Activetime));
Basis = zeros(4,numel(Time));
for k = 1:4
    Basis(k,Activetime) = sin(2*pi*f*(Time(Activetime))).*exp(-((Time(Activetime)-tao(k))/omega(k)).^2);
end
Basis = Basis./repmat(sqrt(sum(Basis.^2,2)),1,size(Basis,2));
% Basis = (orth(Basis'))';
AA =  [1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    1 0 0 0;
    1 0 0 0;
    1 0 0 0];
%        AA = eye(4)*A;
ActiveVox = [];
for k = 1:numel(seedvox)
    s_real(ActiveVoxSeed{k},:) = repmat(AA(k,:)*Basis,numel(ActiveVoxSeed{k}),1);
    ActiveVox = union(ActiveVoxSeed{k},ActiveVox);
end
s_real = s_real ./ norm(s_real, 'fro');   % 归一化
% ===============add pinknoise=======================%
flag = 1;
while flag
    ind = randperm(nSource); 
    %5个大小为3~5cm^2面积的区域，作为5个cluster的面积。
    %注意：spm中计算的Cortex.VertArea是以毫米为单位的,1cm^2= 100mm^2,1个vert所占的面积大概为20mm^2.
    pink_nclus =5;
    pink_Area = unifrnd(3,5,1,pink_nclus)*1e-4;
    %在Vertcies中随机选5个作为种子点
    pink_seedvox = ind(1:pink_nclus);
    pink_ActiveVoxSeed = num2cell(pink_seedvox);
    pink_ActiveVox = [];
     for k = 1:numel(pink_seedvox)
            pink_ActiveVoxSeed{k} = PatchGenerate(pink_seedvox(k),Cortex.VertConn,VertArea,pink_Area(k));%在seedvod处生成大小为DefinedArea的patch
            pink_ActiveVox = union(pink_ActiveVoxSeed{k},pink_ActiveVox);%将5个patch合并到一个列表，共181个网格。
     end
     flag = numel(intersect(pink_ActiveVox,ActiveVox));
end
%pn为活跃vertices的时间过程
pn = mkpinknoise(numel(Time),numel(pink_ActiveVox))';  % 
%转换脑源噪声,并且归一化处理
EEG_brain_noise = zeros(nSource,numel(Time)); % 
EEG_brain_noise(pink_ActiveVox,:) = pn;
EEG_brain_noise = EEG_brain_noise ./ norm(EEG_brain_noise,'fro');
%==================================脑源信号+脑源噪声，映射到头皮，转换为传感器信号=============================
ri = 1/10^(SNIR/10);    % SNIR = 10lg(Ps/Pn)
EEG_brain_signal = s_real + ri*EEG_brain_noise;
EEG_sensor_signal = Gain*EEG_brain_signal;
% =======================================================================%
% %==================== White Gaussian Noise ============================= %
B = awgn(EEG_sensor_signal,SNR,'measured');
%%
% ======================================%
%         Leadfield Matrix normalization
% % =====================================%
LfvW = (mean(Gain.^2,1)).^0.5;
Gain = Gain.*kron(ones(nSensor,1),1./LfvW);

%% Whiten measurements and lead field matrix
Bpre = B(:,1:StimTime);
Cov_n = (Bpre - repmat(mean(Bpre,2),1,StimTime)) * (Bpre - repmat(mean(Bpre,2),1,StimTime))'/(StimTime - 1);
rnkC_noise = rank(single(Cov_n));
variance = diag(Cov_n);
isPca = 1;
if isequal(Cov_n, diag(variance))
    isPca = 0;
end
[V,D] = eig(Cov_n);
D = diag(D);
[D,I] = sort(D,'descend');
V = V(:,I);
if ~isPca
    D = 1./D;
    W = diag(sqrt(D)) * V';
else
    D = 1 ./ D;
    D(rnkC_noise+1:end) = 0;
    W = diag(sqrt(D)) * V';
    W = W(1:rnkC_noise,:);
end
clear V D I
L = W*Gain;
B = W*B;
end


function noise= mkpinknoise(n,m)
% makes m channels of pink noise of length n. Each column is one channel
%
% Guido Nolte, 2012-2015
% g.nolte@uke.de

% If you use this code for a publication, please ask Guido Nolte for the
% correct reference to cite.

% License
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see http://www.gnu.org/licenses/.

% randn('state',sum(100*clock))
n1=2*ceil(n/2)+1;
scal=sqrt(1./[1:(n1-3)/2]');
ff=zeros(n1,m);
ff(2:(n1-1)/2,:)=repmat(scal,1,m);
noise=fft(randn(n1,m));
noise=2*real(ifft(noise.*ff));
noise=noise(1:n,:);

end
