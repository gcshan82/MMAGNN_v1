import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
from torch.utils.data import random_split
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import pickle
from torch_geometric import datasets
import json
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, Dataset
import os
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, CGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from pandas import DataFrame as df
import glob
import csv
# from data import GraphDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
import argparse
import copy
import torch_geometric.transforms as T
from torch_geometric.nn import DynamicEdgeConv
from typing import Optional
# def load_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", default=-1, type=int)
#     args = parser.parse_args()
#     args.local_rank = torch.device("cuda")



class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 50,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


class StructureDataset_large(Dataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    print('DDP set success')

class OutputFusion(nn.Module):
    def __init__(self, feature_dim=64):
        super(OutputFusion, self).__init__()
        # 可训练权重，初始化为均匀分布
        self.weights = nn.Parameter(torch.randn(2, feature_dim))

    def forward(self, output1, output2):
        """
        output1, output2: Tensor, shape=(num, 64)
        """

        normalized_weights = F.softmax(self.weights, dim=0)  # shape=(2, 64)

        fused_output = (
                normalized_weights[0].unsqueeze(0) * output1 +
                normalized_weights[1].unsqueeze(0) * output2
        )

        return fused_output


class AttentionFusion(nn.Module):
    def __init__(self, in_dim: int = 64):
        super(AttentionFusion, self).__init__()
        self.att = nn.Linear(2 * in_dim, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        x = torch.cat([x1, x2], dim=-1)

        scores = self.att(x)

        alpha = F.softmax(scores, dim=-1)


        alpha1 = alpha[..., 0].unsqueeze(-1)
        alpha2 = alpha[..., 1].unsqueeze(-1)

        fused = alpha1 * x1 + alpha2 * x2
        return fused, alpha



# CGCNN
class CGCNN(torch.nn.Module):
    def __init__(
            self,
            # data,
            num_node_feature = 9,
            # num_node_feature = 114,
            dim1=64,
            dim2=64,
            pre_fc_count=1,
            gc_count=3,
            post_fc_count=1,
            pool="global_mean_pool",
            pool_order="early",
            batch_norm="True",
            batch_track_stats="True",
            act="relu",
            dropout_rate=0.5,
            Fusiontype = 'fusion',
            **kwargs
    ):
        super(CGCNN, self).__init__()

        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.k = 20
        self.weights = nn.Parameter(torch.randn(2, 64))
        self.Fusiontype = Fusiontype
        ##Determine gc dimension dimension

        if pre_fc_count == 0:
            gc_dim = num_node_feature

        else:
            gc_dim = dim1

        if pre_fc_count == 0:
            # post_fc_dim = data.num_features
            post_fc_dim = num_node_feature
        else:
            post_fc_dim = dim1

        output_dim = 1
        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin = 0,
                vmax = 8.0,
                bins = 50
            ))
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(num_node_feature, dim1)
                    # lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()
        self.pre_lin_down = torch.nn.Linear(3, 1)
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):

            conv = CGConv(
                gc_dim, 50 , aggr="mean", batch_norm=False
            )
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)

                ##Set up set2set pooling (if used)
        ##Should processing_setps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)

            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)
        self.dgconv1 = DynamicEdgeConv(nn = nn.Sequential(
            nn.Linear(2 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()), k = self.k)

        self.dgconv2 = DynamicEdgeConv(nn = nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()), k = self.k)

        self.dgconv3 = DynamicEdgeConv(nn = nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()), k = self.k)

        self.dgconv4 = DynamicEdgeConv(nn = nn.Sequential(
            nn.Linear(2 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()), k = self.k)
        self.dgbn1 = nn.BatchNorm1d(64)
        self.dgbn2 = nn.BatchNorm1d(128)
        self.dgbn3 = nn.BatchNorm1d(256)
        self.lin_dgcnn = nn.Sequential(
            nn.Linear(448, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())
    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                # out = self.pre_lin_down(data.x.float())
                # out = getattr(F, self.act)(out)
                # out = out.squeeze()
                # out = self.pre_lin_list[i](out)
                data_Normal= T.NormalizeFeatures()(data)
                out = self.pre_lin_list[i](data_Normal.x)
                # out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)  # 激活函数
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)
        attr = self.edge_embedding(data.edge_weight)
        # attr = data.edge_attr
        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    # out = self.conv_list[i](data.x, data.subgraph_edge_index, data.subgraph_edge_attr)
                    out = self.conv_list[i](data.x, data.edge_index, attr)
                    out = self.bn_list[i](out)
            else:
                if self.batch_norm == "True":
                    # out = self.conv_list[i](out, data.subgraph_edge_index, data.subgraph_edge_attr)
                    out = self.conv_list[i](out, data.edge_index, attr)
                    out = self.bn_list[i](out)

            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            ##DGCNN layers
            pos, batch = data.pos, data.batch

            x1 = self.dgconv1(pos, batch)
            x1 = self.dgbn1(x1)
            # print('layer1 num:', x1.shape)
            x2 = self.dgconv2(x1, batch)
            x2 = self.dgbn2(x2)
            # print('layer2 num:', x2.shape)
            x3 = self.dgconv3(x2, batch)
            x3 = self.dgbn3(x3)

            position = torch.cat((x1, x2, x3), dim = 1)
            position = self.lin_dgcnn(position)
            if self.Fusiontype == 'attention':
                attention_fusion_model = AttentionFusion(feature_dim = 64)
                out = attention_fusion_model(out,position)
                # fusion_type = OutputFusion(feature_dim = 64)


            # 融合输出
            else:
                normalized_weights = F.softmax(self.weights, dim=0)  # shape=(2, 64)
                out = (
                        normalized_weights[0].unsqueeze(0) * out +
                        normalized_weights[1].unsqueeze(0) * position
                )


            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

            if out.shape[1] == 1:
                return out.view(-1)
            else:
                return out


def train(model, crit, optimizer, train_loader, device):
    loss_all = 0
    count = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        count = count + output.size(0)
    loss_all = loss_all / count

    # return loss_all/len(dataset)
    return loss_all

def eval(model, loader, criterion, device, split='Val'):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            data = data.to(device)

            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('{} loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
        split, epoch_loss, epoch_mse, epoch_mae, toc - tic))
    return epoch_mae, epoch_mse

def main_worker(rank, w_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = os.getenv('LOCAL_RANK', -1), type = int)
    args = parser.parse_args()
    ddp_setup(rank, 4)

    device = torch.device('cuda')
    # dataset = StructureDataset_large('./hmof_sample/mof_25_train_all_para')
    dataset = StructureDataset('./hmof_sample/mof_train_pos_25_noneattr')
    # datast = GraphDataset(dataset, degree=True, k_hop=1, se='khopgnn',
    #                               use_subgraph_edge_attr = True)
    print(dataset[13])
    train_len = int(0.9 * len(dataset))
    valid_len = len(dataset) - train_len
    # valid_len = int(0.1 * len(dataset))
    # test_len = len(dataset) - train_len - valid_len
    # data, slices = StructureDataset.collate(datast)
    # torch.save((data, slices), os.path.join('./test_mof_sub_25_attr/processed/data.pt'))
    seed_num = 1
    torch.manual_seed(seed_num)

    train_dataset, val_dataset = random_split(dataset, [train_len, valid_len])
    print(len(train_dataset))
    print(len(val_dataset))

    # train_dataset = dataset[0:1500]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size = 64,
                              sampler = train_sampler)
    # val_dataset = dataset[1500:]
    val_loader = DataLoader(val_dataset, batch_size = 64,
                              shuffle=False)
    model = CGCNN().to(device)

    OutputFusion().to(device)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids = [args.local_rank], find_unused_parameters = True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay = 1e-3)
    crit = torch.nn.MSELoss()


    # 如果GPU可用用GPU，否则用CPU

    # 训练
    loss_list = []
    eval_MAE_list = []
    eval_MSE_list = []
    epoch_num = 200
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    for epoch in range(epoch_num):
        print('epoch', epoch)
        tic = timer()
        loss = train(model, crit, optimizer, train_loader, device)

        # loss = loss.item()
        loss_list.append(loss)
        toc = timer()

        print('Train loss: {:.4f} time: {:.2f}s'.format(
            loss, toc - tic))
        eval_MAE,  eval_MSE= eval(model, val_loader, crit, device)
        eval_MAE_list.append(eval_MAE)
        eval_MSE_list.append(eval_MSE)
        if eval_MAE < best_val_loss:
            best_val_loss = eval_MAE
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)
    print(loss_list)
    print(eval_MAE_list)
    print(eval_MSE_list)
    # 打开一个文本文件以写入模式
    outdir = './cgcnn_dgnn_record/epoch_'+str(epoch_num)+'_seed_'+str(seed_num)+'_num_'+str(90000)+'_dim_'+str(64)+'_dropout_'+str(0.3)+ '_decay_'+str(0.001)+ '_loss_MSE'+'_Normalize'+'CO2'+'_lr_'+str(0.0001)

    # os.makedirs(outdir)
    # outdir = args.outdir
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except Exception:
            pass
    with open(outdir + "/loss.txt", "w") as file:
        # 遍历列表中的元素并将它们写入文件
        for item in loss_list:
            file.write(str(item) + "\n")
    with open(outdir +"/eval_MAE.txt", "w") as file:
        # 遍历列表中的元素并将它们写入文件
        for item in eval_MAE_list:
            file.write(str(item) + "\n")
    with open(outdir +"/eval_MSE.txt", "w") as file:
        # 遍历列表中的元素并将它们写入文件
        for item in eval_MSE_list:
            file.write(str(item) + "\n")

    if dist.get_rank() == 0:
        torch.save({'state_dict': model.state_dict()}, outdir +'/sample_model.pt')


# dataset = StructureDataset_large('./test_mof_large')
# print(dataset[13])
# print('x:')
# print(dataset[13].x)
# print('edge_index:')
# print(dataset[13].edge_index)
# print('y:')
# print(dataset[13].y)
# print('structure_id:')
# print(dataset[13].structure_id)
# print('edge_weight:')
# print(dataset[13].edge_weight)
# print('edge_attr:')
# print(dataset[13].edge_attr)
#
# dataset = StructureDataset_large('./test_mof_large_2.5')
# print(dataset[13])
# print('x:')
# print(dataset[13].x)
# print('edge_index:')
# print(dataset[13].edge_index)
# print('y:')
# print(dataset[13].y)
# print('structure_id:')
# print(dataset[13].structure_id)
# print('edge_weight:')
# print(dataset[13].edge_weight)
# print('edge_attr:')
# print(dataset[13].edge_attr)

def main():
    # args = parser.parse_args()
    # args.nprocs = torch.cuda.device_count()
    global args
    mp.spawn(main_worker, nprocs=4, args=(4,))

if __name__ == "__main__":
    main()

