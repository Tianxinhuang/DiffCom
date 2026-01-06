import random

import numpy as np
import torch
from pointnet2_ops.pointnet2_utils import QueryAndGroup, gather_operation, furthest_point_sample
from torch.utils.data import DataLoader, Dataset


def fpsfunc(points, num):
    fidx = furthest_point_sample(points[:, :, :3].contiguous(), num)
    fgts = gather_operation(points.permute(0, 2, 1).contiguous(), fidx)
    return fgts


class GTDataset(Dataset):
    def __init__(self, double_layer_tensor_list, level=None, target_size=4096):
        self.double_layer_tensor_list = double_layer_tensor_list
        self.target_size = target_size
        if level is None:
            self.flat_list = [
                (outer_idx, inner_idx, tensor)
                for outer_idx, inner_list in enumerate(double_layer_tensor_list)
                for inner_idx, tensor in enumerate(inner_list)
            ]
        else:
            self.flat_list = [
                (level, inner_idx, tensor)
                for inner_idx, tensor in enumerate(double_layer_tensor_list[level])
            ]

    def __len__(self):
        return len(self.flat_list)

    def __getitem__(self, idx):
        outer_idx, inner_idx, tensor = self.flat_list[idx]
        num_points = tensor.shape[0]
        if num_points > self.target_size:
            indices = torch.randperm(num_points)[:self.target_size]
            sampled_tensor = tensor[indices]
        else:
            indices = torch.randint(0, num_points, (self.target_size,))
            sampled_tensor = tensor[indices]
        return sampled_tensor, (outer_idx, inner_idx)


def gt_collate_fn(batch):
    tensors, all_indices = zip(*batch)
    outer_inner_indices, point_indices = zip(*[(outer_idx, inner_idx) for outer_idx, inner_idx in all_indices])
    return torch.stack(tensors, dim=0), outer_inner_indices, point_indices


def define_gtloader(double_layer_tensor_list, tsize=4096, bsize=8):
    dataset = GTDataset(double_layer_tensor_list, target_size=tsize)
    dataloader = DataLoader(dataset, batch_size=bsize, collate_fn=gt_collate_fn, shuffle=True)
    return dataloader


class CenDataset(Dataset):
    def __init__(self, tensor_list, target_size=4096):
        self.tensor_list = tensor_list
        self.target_size = target_size

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        tensor = self.tensor_list[idx]
        num_points = tensor.shape[0]
        if num_points > self.target_size:
            indices = torch.randperm(num_points)[:self.target_size]
            tensor = tensor[indices]
        else:
            indices = torch.randint(0, num_points, (self.target_size,))
            tensor = fpsfunc(tensor.unsqueeze(0), self.target_size).squeeze(0)
        return tensor


def cen_collate_fn(batch):
    return torch.stack(batch, dim=0)


def define_cenloader(single_layer_tensor_list, fpsfunc, tsize=1024, bsize=8):
    dataset = CenDataset(single_layer_tensor_list, fpsfunc, target_size=tsize)
    dataloader = DataLoader(dataset, batch_size=bsize, collate_fn=cen_collate_fn, shuffle=False)
    return dataloader


def iter_loader(dataloader):
    result = []
    for _, batch in enumerate(dataloader):
        result.append(batch)
    result = torch.cat(result, dim=0)
    return result


def get_loader(dataloader, tensorlist, iternum=100):
    result = []
    data_iter = iter(dataloader)
    j = 0
    for _ in range(iternum):
        try:
            batch, depthidx, pointidx = next(data_iter)
            cens = get_cen(tensorlist, depthidx, pointidx)
            # print(cens.shape)
            # assert False
            result.append(batch)
        except StopIteration:
            print('one stop')
            j += 1
            data_iter = iter(dataloader)
            batch, depthidx, pointidx = next(data_iter)
            result.append(batch)
    result = torch.cat(result, dim=0)
    return result


def get_info(cens, boxes, depthidx, pointidx):
    didx = np.array(depthidx, dtype=np.int32)
    pidx = np.array(pointidx, dtype=np.int32)
    result = []
    boxres = []
    num = len(didx)
    for i in range(num):
        result.append(cens[didx[i]][pidx[i]].unsqueeze(0))
        boxres.append(boxes[didx[i]][pidx[i]].unsqueeze(0))
    result = torch.cat(result, dim=0)
    boxres = torch.cat(boxres, dim=0)
    return result, boxres


if __name__ == "__main__":
    random.seed(42)
    double_layer_tensor_list = [
        [torch.rand(random.randint(1, 10000), 3) for _ in range(random.randint(1, 5))]
        for _ in range(4)
    ]
    tensor_list = [torch.rand(random.randint(1, 10000), 3) for _ in range(21)]
    gtloader = define_gtloader(double_layer_tensor_list)
    result = get_loader(gtloader, tensor_list)
    print(result.shape)