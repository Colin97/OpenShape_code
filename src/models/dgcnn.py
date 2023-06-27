import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch_redstone as rst


def knn(x, k):
    x = x.transpose(2, 1)
    idx = rst.Polyfill.cdist2(x, x).topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class NoCuDNNBatchNorm(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        torch.backends.cudnn.enabled = False
        x = self.bn(x)
        torch.backends.cudnn.enabled = True
        return x


class NoCuDNNBatchNorm1d(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        torch.backends.cudnn.enabled = False
        x = self.bn(x)
        torch.backends.cudnn.enabled = True
        return x


class DGCNN(nn.Module):
    def __init__(self, in_dim=3, repr_dim=256, scaling=1):
        super().__init__()
        self.n_knn = 20

        base_size = int(64 * scaling)
        self.bn1 = NoCuDNNBatchNorm(base_size)
        self.bn2 = NoCuDNNBatchNorm(base_size)
        self.bn3 = NoCuDNNBatchNorm(base_size * 2)
        self.bn4 = NoCuDNNBatchNorm(base_size * 4)
        self.bn5 = NoCuDNNBatchNorm1d(base_size * 16)

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim * 2, base_size, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(base_size*2, base_size, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(base_size*2, base_size*2, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(base_size*4, base_size*4, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv1d(base_size*8, base_size*16, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.linear1 = nn.Linear(base_size*32, base_size*8, bias=False)
        self.bn6 = nn.BatchNorm1d(base_size*8)
        self.dp1 = nn.Dropout(p=0.0)
        self.linear2 = nn.Linear(base_size*8, repr_dim)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, device=None, quantization_size=0.05):
        x = features.transpose(-1, -2)
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        x = self.linear2(x)
        return x


def make(cfg):
    return DGCNN(cfg.model.in_channel, cfg.model.out_channel, cfg.model.scaling)
