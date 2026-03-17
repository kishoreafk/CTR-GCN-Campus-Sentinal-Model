"""
CTR-GCN backbone for skeleton-based action recognition.
Input:  (N, C=3, T=64, V=18, M=2)
Output: (N, 256) feature vector

Architecture: 10 ST-GCN blocks with increasing channels:
  3→64→64→64→64→128→128→128→256→256→256
Temporal downsampling at blocks 4 and 7 (stride=2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ctrgcn.layers import STGCNBlock


class CTRGCN(nn.Module):
    def __init__(self, graph, in_channels: int = 3,
                 base_channels: int = 64, num_classes: int = 0):
        super().__init__()

        # Register adjacency as buffer (not a parameter)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer('A', A)
        self.num_nodes = graph.num_nodes
        num_subsets = A.shape[0]

        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

        # 10 ST-GCN blocks
        # Channels: 3→64, 64→64, 64→64, 64→64(stride2),
        #           64→128, 128→128, 128→128(stride2),
        #           128→256, 256→256, 256→256
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, base_channels, num_subsets,
                       self.num_nodes, residual=False),          # 0
            STGCNBlock(base_channels, base_channels, num_subsets,
                       self.num_nodes),                          # 1
            STGCNBlock(base_channels, base_channels, num_subsets,
                       self.num_nodes),                          # 2
            STGCNBlock(base_channels, base_channels, num_subsets,
                       self.num_nodes, stride=2),                # 3
            STGCNBlock(base_channels, base_channels * 2, num_subsets,
                       self.num_nodes),                          # 4
            STGCNBlock(base_channels * 2, base_channels * 2, num_subsets,
                       self.num_nodes),                          # 5
            STGCNBlock(base_channels * 2, base_channels * 2, num_subsets,
                       self.num_nodes, stride=2),                # 6
            STGCNBlock(base_channels * 2, base_channels * 4, num_subsets,
                       self.num_nodes),                          # 7
            STGCNBlock(base_channels * 4, base_channels * 4, num_subsets,
                       self.num_nodes),                          # 8
            STGCNBlock(base_channels * 4, base_channels * 4, num_subsets,
                       self.num_nodes),                          # 9
        ])

        self.out_channels = base_channels * 4  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V, M)
        Returns: (N, 256) feature vector
        """
        N, C, T, V, M = x.shape

        # Merge person dimension into batch
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)                   # (N*M, C, T, V)

        # Input normalization
        x_bn = x.permute(0, 2, 3, 1).contiguous()    # (N*M, T, V, C)
        x_bn = x_bn.view(N * M, T, V * C)             # (N*M, T, V*C)
        x_bn = x_bn.permute(0, 2, 1).contiguous()     # (N*M, V*C, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # x is now (N*M, C, T, V)

        # Forward through ST-GCN blocks
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # Global average pooling over T and V
        x = x.mean(dim=[2, 3])  # (N*M, C_out)

        # Reshape and pool over persons
        x = x.view(N, M, -1).mean(dim=1)  # (N, C_out)

        return x
