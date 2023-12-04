import torch
from torch import nn
import torch.nn.functional as F

from utils.dgcnn_utils import get_graph_feature

class DGCNN_cls(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512,512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bin_num = [1, 2, 4, 8, 16, 32]
        self.transform = nn.Linear(63, 1)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, args.latent_dim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        self.fc_bn3_m = nn.BatchNorm1d(512)
        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, args.latent_dim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)


    def forward(self, x):
        batch_size = x.size(0)
        x=x.permute(0,2,1)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)


        batch, feat_dim, n_points = x.shape
        bin_feat = []
        for bin in self.bin_num:
           z = x.view(batch, feat_dim, bin, -1)
           z_max, _ = z.max(3)
           z = z.mean(3) + z_max
           bin_feat.append(z)
        bin_feat = torch.cat(bin_feat, 2).contiguous()  # (batchsize, 1024, 31)    b 512 63
        bin_feat = self.transform(bin_feat).squeeze()# b 512
        return bin_feat   #(B 512)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 512, num_points) -> (batch_size, 512)

        # # Mapping to [c], cmean
        # m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        # m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        # m = self.fc3_m(m)
        # v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        # v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        # v = self.fc3_v(v)







# if __name__ == '__main__':
#
#     point = torch.rand((5, 3, 128))
#     output = model(point)
#     print(output.shape)