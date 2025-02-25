from torch.nn import Conv2d, Parameter, Softmax
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple

class DWConv(nn.Module):  #DWCONV
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, stride=1):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DWConv_depth(nn.Module):  #DWCONV
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dilation=1, stride=1):
        super(DWConv_depth, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, groups=in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class PWConv(nn.Module):  #PWCONV
    def __init__(self, in_channels, out_channels,padding=0, dilation=1, stride=1):
        super(PWConv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pointwise(x)
        return x



class PConv_2(nn.Module):    #简单两两分开
    def __init__(self, dim: int, n_div: int, forward: str = "split_cat", kernel_size: int = 3) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.pwconv = PWConv(512, 512, 1, 1)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError


    def forward_split_cat(self, x: Tensor) -> Tensor:
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x3 = self.pwconv(x1)
        return x3, x2




class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):

        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        #self.p_conv.register_backward_hook(self._set_lr)   #不完全钩子
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod   #静态方法
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset







class GroupedMaxPool2d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):

        super(GroupedMaxPool2d, self).__init__()
        self.groups = groups
        self.grouped_pools = nn.ModuleList()

        # 每个组的核大小
        self.kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
        # 每个组的步幅
        self.strides = [(2, 2)] * groups
        # 每个组的填充大小
        self.paddings = [((k[0] // 2) - 1, (k[1] // 2) - 1) for k in self.kernel_sizes]

        assert len(self.kernel_sizes) == groups
        assert len(self.strides) == groups
        assert len(self.paddings) == groups

        for i in range(groups):
            self.grouped_pools.append(
                nn.MaxPool2d(kernel_size=self.kernel_sizes[i], stride=self.strides[i], padding=self.paddings[i]))

    def forward(self, x):
        split_size = x.size(1) // self.groups
        x_split = torch.split(x, split_size, dim=1)
        out = [pool(x_split[i]) for i, pool in enumerate(self.grouped_pools)]
        return torch.cat(out, dim=1)


class AvgMaxPooling(nn.Module):
    def __init__(self, kernel_size):
        """
        Args:
            kernel_size (int): 池化核大小
        """
        super(AvgMaxPooling, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        return avg_pooled, max_pooled



from models.ERF import asyConv
class PAFEM(nn.Module):              #普通卷积加FasterNet
    def __init__(self, dim,in_dim,device=None):   #dim = 2048  in_dim = 2048
        super(PAFEM, self).__init__()
        self.down_conv = nn.Sequential(
            DWConv(2048, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )


        down_dim = in_dim // 2       #down_dim = 1024

        self.pconv_2 = PConv_2(dim=1024, n_div=2, forward="split_cat", kernel_size=3)
        self.pwconv = PWConv(1024,1024,1,1)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()

        self.asyconv = asyConv(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')


        self.deform_conv = DeformConv2d(3 * down_dim, down_dim, 3, 1, 1, None, True)

        self.conv1 = nn.Sequential(       #  通道//2
            DWConv(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU()
        )

        self.conv2 = nn.Sequential(       #  通道//2   dilation=2
            DWConv(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.ReLU()
        )

        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = Parameter(torch.zeros(1))  #一个可学习的标量参数，初始值为 0。

        self.conv3 = nn.Sequential(     #通道//2    dilation=3
            DWConv(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = Parameter(torch.zeros(1))


        self.conv4 = nn.Sequential(    #通道//2    dilation=6
            DWConv(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(     #通道//2
            DWConv(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(      #通道*2.5
            DWConv(3 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.ReLU()
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, x):  #4,2048,24,24

        conv2_asy = self.asyconv(x)  # 水平和垂直卷积

        #conv1
        conv1 = self.conv1(x)   #2048

        # conv2
        conv2 = self.conv2(x)
        conv2 = torch.cat((conv2, conv2_asy), 1)
        conv2 = self.down_conv(conv2)

        m_batchsize, C, height, width = conv2.size()    #4,512,24,24
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)

        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)

        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2* out2 + conv2
        out2_1,out2_2 =self.pconv_2(out2)

        # conv3
        conv3 = self.conv3(x)
        conv3 = torch.cat((conv3, conv2_asy), 1)
        conv3 = self.down_conv(conv3)

        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)

        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)

        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        out3_1, out3_2 = self.pconv_2(out3)

        #conv4
        conv4 = self.conv4(x)
        conv4 = torch.cat((conv4, conv2_asy), 1)
        conv4 = self.down_conv(conv4)

        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)

        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)

        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        out4_1, out4_2 = self.pconv_2(out4)   #512

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')

        outA_D = out2_1 + out3_2   #512
        outA_F = out2_1 + out4_2

        outC_B = out3_1 + out2_2
        outC_F = out3_1 + out4_2

        outE_B = out4_1 + out2_2
        outE_D = out4_1 + out3_2

        outA = self.relu(self.bn(self.pwconv((torch.cat((outA_D,outA_F),1)))))   #1024
        outC = self.relu(self.bn(self.pwconv((torch.cat((outC_B,outC_F),1)))))
        outE = self.relu(self.bn(self.pwconv((torch.cat((outE_B,outE_D),1)))))


        out2 = outA + out2   #1024
        out3 = outC + out3
        out4 = outE + out4

        out2 = self.relu(self.bn((out2)))
        out3 = self.relu(self.bn((out3)))
        out4 = self.relu(self.bn((out4)))

        deform_output = self.deform_conv(torch.cat((out2,out3,out4),1))
        add_all = torch.cat((conv1,deform_output, conv5), 1)

        Fuse_feature = self.fuse(add_all)

        return Fuse_feature



