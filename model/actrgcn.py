import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        graph = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))
        graph = self.conv4(graph)
        graph_c = graph * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        y = torch.einsum('ncuv,nctv->nctu', graph_c, x3)
        return y, graph

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        graph_list = []
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z, graph = self.convs[i](x, A[i], self.alpha)
            graph_list.append(graph)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y, torch.stack(graph_list, 1)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        # self.tcn1 = TemporalConv(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        z, graph = self.gcn1(x)
        y = self.relu(self.tcn1(z) + self.residual(x))
        return y, graph

    
class ChannelAttention_p(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_p, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)#nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # avg_spa = torch.mean(SCA_3_o1, dim=1, keepdim=True)
        # max_spa, _ = torch.max(SCA_3_o1, dim=1, keepdim=True)
        # SCA_3_o2 = torch.cat([avg_out, max_out], dim=1)
        # SCA_3_o2 = self.spatial_attention_rgb(SCA_3_o2)
        out = max_out+avg_out
        return self.sigmoid(out)   
class ChannelAttention_m(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_m, self).__init__()

        #self.avg_pool = nn.AdaptiveAvgPool2d(1)#nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        # self.spatial_attention=nn.Sequential(
        #     nn.Conv2d(2, 1, 7, padding=3, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
    
    

class fusion(nn.Module):
    def __init__(self,T_in_channels):
        super(fusion,self).__init__()
        self.att_T_p=ChannelAttention_p(T_in_channels)
        self.att_N_p=ChannelAttention_p(16)
        self.att_T_m = ChannelAttention_m(T_in_channels)
        self.att_N_m = ChannelAttention_m(16)
        
        
        # self.att_T_p=ChannelAttention(T_in_channels)
        # self.att_N_p=ChannelAttention(16)
        # self.att_T_m = ChannelAttention(T_in_channels)
        # self.att_N_m = ChannelAttention(16)
        
    def forward(self,x_p,x_m):
        #B*C*T*N 自适应融合，T和N各自轴上
        #情感特征的参数化，或者平均方式的参数化。
        B,C,T,N=x_p.size()
        x_p_T=x_p.permute(0,2,1,3)
        x_p_N=x_p.permute(0,3,2,1)
        x_m_T = x_m.permute(0, 2, 1, 3)
        x_m_N = x_m.permute(0, 3, 2, 1)


        att_N_p_map = (self.att_N_p(x_p_N)).permute(0, 3, 2, 1)
        x_p_mid = (x_p+x_p * att_N_p_map).permute(0, 2, 1, 3)
        att_T_p_map=(self.att_T_p(x_p_mid)).permute(0,2,1,3)

        att_N_m_map = (self.att_N_m(x_m_N)).permute(0, 3, 2, 1)
        x_m_mid = (x_m+x_m * att_N_m_map).permute(0, 2, 1, 3)
        att_T_m_map = (self.att_T_m(x_m_mid)).permute(0, 2, 1, 3)



        x_p=x_p+x_m*att_T_m_map
        x_m=x_m+x_p*att_T_p_map

        # x_p=x_p+x_m*att_T_m_map*att_N_m_map
        # x_m=x_m+x_p*att_T_p_map*att_N_p_map
        # x_p = x_p + x_m * (att_T_m_map+att_N_m_map)
        # x_m = x_m + x_p * (att_T_p_map+att_T_p_map)

        return x_p,x_m   
    
    

class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_person=2, num_constraints=31,graph=None, graph_args=dict(), in_channels_p=3,in_channels_m=8,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        base_channel = 64
        self.l1_p = TCN_GCN_unit(in_channels_p, base_channel, A, residual=False, adaptive=adaptive)
        self.l2_p = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3_p = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4_p = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5_p= TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6_p = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7_p = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8_p = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9_p = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10_p = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        self.l1_m = TCN_GCN_unit(in_channels_m, base_channel, A, residual=False, adaptive=adaptive)
        self.l2_m = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3_m = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4_m = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5_m = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6_m = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7_m = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8_m = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9_m = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10_m = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        
        self.fusion1 = fusion(48)
        self.fusion2 = fusion(24)
        self.fusion3 = fusion(12)
        
        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints*48)
        nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints*48)))
        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)
        # self.fc = nn.Linear(base_channel*4, num_class)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        # bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    
    def partDivison(self, graph):
        _, k, u, v = graph.size() # n k u v
        head = [ 3]
        left_arm = [7,8,9]
        right_arm = [4,5,6]
        torso = [0,1,2]
        left_leg = [ 13, 14, 15]
        right_leg = [10,11,12]
        graph_list = []
        part_list = [head, torso, right_arm, left_arm, right_leg, left_leg]
        for part in part_list:
            part_grah = graph[:,:,part,:].mean(dim=2, keepdim=True)
            graph_list.append(part_grah)
        graph = torch.cat(graph_list, 2)
        graph_list = []
        for part in part_list:
            part_grah = graph[:,:,:,part].mean(dim=-1, keepdim=True)
            graph_list.append(part_grah)
        return torch.cat(graph_list, -1)

    def forward(self, x_p, x_m):
        # if len(x_p.shape) == 3:
        #     N, T, VC = x.shape
        #     x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        # N, C, T, V, M = x.size()
        
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()
        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_m = x_m.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_m, T)


        x_p = self.data_bn_p(x_p)
        x_m = self.data_bn_m(x_m)

        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_m = x_m.view(N, M, V, C_m, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_m, T, V)

        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x_p, _ = self.l1_p(x_p)
        x_p, _ = self.l2_p(x_p)
        x_p, _ = self.l3_p(x_p)
        x_p, _ = self.l4_p(x_p)
        x_m, _ = self.l1_m(x_m)
        x_m, _ = self.l2_m(x_m)
        x_m, _ = self.l3_m(x_m)
        x_m, _ = self.l4_m(x_m)
        
        x_p,x_m=self.fusion1(x_p,x_m)
        
        
        x_p, _ = self.l5_p(x_p)
        x_p, _ = self.l6_p(x_p)
        x_p, _ = self.l7_p(x_p)
        x_m, _ = self.l5_m(x_m)
        x_m, _ = self.l6_m(x_m)
        x_m, _ = self.l7_m(x_m)
        
        
        x_p,x_m=self.fusion2(x_p,x_m)
        
        x_p, _ = self.l8_p(x_p)
        x_p, _ = self.l9_p(x_p)
        x_p, graph = self.l10_p(x_p)
        x_m, _ = self.l8_m(x_m)
        x_m, _ = self.l9_m(x_m)
        x_m, _ = self.l10_m(x_m)
        
        x_p,x_m=self.fusion3(x_p,x_m)
        # x_m, _ = self.l1_m(x_m)
        # x_m, _ = self.l2_m(x_m)
        # x_m, _ = self.l3_m(x_m)
        # x_m, _ = self.l4_m(x_m)
        # x_m, _ = self.l5_m(x_m)
        # x_m, _ = self.l6_m(x_m)
        # x_m, _ = self.l7_m(x_m)
        # x_m, _ = self.l8_m(x_m)
        # x_m, _ = self.l9_m(x_m)
        # x_m, _ = self.l10_m(x_m)

        # N*M,C,T,V
        
        c_new_m = x_m.size(1)
        x_m = x_m.view(N, M, c_new_m, -1)
        x_m = x_m.mean(3).mean(1)
        x_m = self.drop_out(x_m)
        
        c_new_p = x_p.size(1)
        x_p = x_p.view(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)
        x_p = self.drop_out(x_p)
        
        c_new = x_p.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)
        # x = self.drop_out(x)
        graph2 = graph.view(N, M, -1, c_new, V, V)
        # graph4 = torch.einsum('n m k c u v, n m k c v l -> n m k c u l', graph2, graph2)
        graph2 = graph2.view(N, M, -1, c_new, V, V).mean(1).mean(2).view(N, -1)
        # graph4 = graph4.view(N, M, -1, c_new, V, V).mean(1).mean(2).view(N, -1)
        # graph = torch.cat([graph2, graph4], -1)
        return self.fc1_classifier_p(x_p),self.fc2_aff(x_p),self.fc1_classifier_m(x_m), graph2