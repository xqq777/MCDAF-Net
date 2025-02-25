import torch
import torch.nn as nn
from torch import Tensor
def Conv1x1(nin, nf, stride=1):
  return nn.Sequential(
      nn.Conv2d(nin, nf, 3, stride, 1, bias=False),
      nn.BatchNorm2d(nf),
      nn.ReLU(inplace=True)
  )


class PConv(nn.Module):
    def __init__(self, dim: int, n_div: int, forward: str = "split_cat", kernel_size: int = 3) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    def forward_slicing(self, x: Tensor) -> Tensor:
        """ Apply forward pass for inference. """
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x
    def forward_split_cat(self, x: Tensor) -> Tensor:
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), 1)
        return x



class PWConv(nn.Module):  #DWCONV
    def __init__(self, in_channels, out_channels,padding=0, dilation=1, stride=1):
        super(PWConv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pointwise(x)
        return x




class PConv_2(nn.Module):
    def __init__(self, dim: int, n_div: int, forward: str = "split_cat", kernel_size: int = 3) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.pwconv = PWConv(1024, 1024, 1, 1)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    def forward_split_cat(self, x: Tensor) -> Tensor:
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.pwconv(x1)
        return x1,x2



class FasterNetBlock(nn.Module):
    def __init__(self, inp: int, outp: int, dim: int, n_div: int, forward: str = "split_cat", kernel_size: int = 3) -> None:
        super().__init__()
        self.pconv = PConv(dim=dim, n_div=n_div, forward=forward, kernel_size=kernel_size)
        self.pwconv = PWConv(inp,outp,1,1)
        self.conv1_1 = Conv1x1(inp, outp)
        self.bn = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        r = self.pconv(r)
        r = self.pwconv(r)
        r = self.bn(r)
        r = self.relu(r)
        r = self.conv1_1(r)
        x = r+x
        return x





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FasterNetBlock(inp=2048,outp=2048, dim=2048,n_div=4).to(device)
#
# # 创建输入张量并将其移至 GPU 上
# batch_size = 4
# input_channels = 2048
# height = 22
# width = 25
# input_tensor = torch.randn(batch_size, input_channels, height, width).to(device)
#
# # 执行前向传播
# output_tensor = model(input_tensor)
# print(output_tensor.shape)
