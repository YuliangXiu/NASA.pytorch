#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, n_elements, n_layers, width, D_dim, structured=False):
        super(ResidualBlock, self).__init__()

        if not structured:
            self.conv_head = nn.Linear(3 * n_elements, width)
            self.conv_tail = nn.Linear(width, D_dim)
        else:
            self.conv_head = nn.Linear(3 + D_dim, width)
            self.conv_tail = nn.Linear(width, 1)

        self.conv = nn.Linear(width, width)
        self.bn = nn.LayerNorm(width)
        self.relu = nn.LeakyReLU(0.1)

        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers-1):
            if i == 0:
                out = self.conv_head(x)
                residual = self.conv_head(x)
            else:
                out = self.conv(out)
            out = self.bn(out)
            out = self.relu(out)
            out += residual

        out = self.conv_tail(out)

        return out
    

class NASANet(nn.Module):
    def __init__(self, n_elements, n_layers, width, D_dim):
        super(NASANet, self).__init__()

        self.n_elements = n_elements
        self.theta = nn.ModuleList()
        self.pi = nn.ModuleList()
        self.softmax = nn.Softmax(dim=2)
        for i in range(self.n_elements):
            self.pi.append(ResidualBlock(n_elements, n_layers, 
                                        width, D_dim, False))
            self.theta.append(ResidualBlock(n_elements, n_layers, 
                                        width, D_dim, True))
        
    def forward(self, data_bx, data_bt):

        # data_BX [B, N, 21, 3]
        # data_BT [B, N, 21, 3]
        # theta_out [B, N, 21]

        B, N = data_bt.shape[:2]
        theta_out = torch.randn(B*N, self.n_elements).cuda()

        for i in range(self.n_elements):
            pi_out = self.pi[i](
                data_bt.reshape(B*N, self.n_elements*3))

            theta_out[:, [i]] = self.theta[i](
                torch.cat((pi_out, data_bx[:,:,i,:].reshape(B*N,3)), dim=1))
        theta_out = theta_out.reshape(B, N, self.n_elements)
        
        return self.softmax(theta_out)


if __name__ == "__main__":
    net = NASANet(21, 4, 40, 4).to("cuda")
    data_bx, data_bt = torch.randn(12, 5000, 21, 3), torch.randn(12, 5000, 21, 3)
    net.forward(data_bx.to("cuda"), data_bt.to("cuda"))