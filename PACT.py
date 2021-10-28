import torch
from torch.nn import functional as F

class PACT_ReLU(torch.autograd.Function):
    """
    def __init__(self, ALPHA):
        super(PACT_ReLU, self).__init__()
        self.alpha = ALPHA
    """

    # Function这里要是静态，就没有init
    # 另外添加静态调用时就不用实例化
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        k = 4  # bitweight of activation
        # print("input:",input)
        # y = 0.5*(torch.abs(input)-torch.abs(input-alpha.item())+alpha.item()) #paper的公式
        y = torch.clamp(input, 0, alpha.item())
        scale = (2 ** k - 1) / alpha.item()
        output = torch.round(y * scale) / scale  # paper的公式

        # return torch.clamp(input, 0, alpha.data)
        # return output
        # print("output:",output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print(ctx.saved_tensors)
        input, alpha, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_alpha = grad_output.clone()
        # print('2. ', grad_input)
        # print('3. ', grad_output.shape)
        # print('4. alpha: ', alpha)

        lower_bound = input < 0
        upper_bound = input > alpha
        x_range = ~(lower_bound | upper_bound)
        # print('x_range:', x_range)
        grad_input[grad_input.le(0)] = 0
        grad_input[grad_input.ge(alpha)] = 0

        # grad_output = torch.zeros_like(grad_output)

        # if grad_input < alpha:
        #    grad_output = 0
        grad_alpha[input.le(alpha)] = 0
        grad_alpha = torch.sum(grad_output * torch.ge(input, alpha).float()).view(-1)

        # return grad_input, torch.sum(grad_alpha), None
        # return grad_input
        # return grad_input, torch.sum(grad_alpha.float()).view(-1), None
        # print("grad_output * x_range.float()",grad_output * x_range.float())
        return grad_output * x_range.float(), grad_alpha, None


class PACT(torch.nn.Module):
    # 扩展module
    def __init__(self, in_features):
        super(PACT, self).__init__()
        self._Alpha = torch.nn.Parameter(torch.tensor(in_features))

        # print('1. ',self.Alpha.shape)

    def forward(self, input):
        # print(self._Alpha)
        return PACT_ReLU.apply(input, self._Alpha)


def quantize_k(r_i, k):
    scale = (2 ** k - 1)
    r_o = torch.round(scale * r_i) / scale
    return r_o


class DoReFaQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i, k):
        tanh = torch.tanh(r_i).float()
        # scale = 2**k - 1.
        # quantize_k = torch.round( scale * (tanh / 2*torch.abs(tanh).max() + 0.5 ) ) / scale
        r_o = 2 * quantize_k(tanh / (2 * torch.max(torch.abs(tanh)).detach()) + 0.5, k) - 1
        # r_o = 2 * quantize_k - 1.
        return r_o

    @staticmethod
    def backward(ctx, dLdr_o):
        # due to STE, dr_o / d_r_i = 1 according to formula (5)
        return dLdr_o, None


class Conv2d(torch.nn.Conv2d):
    def __init__(self, in_places, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 bitwidth=8):
        #这里设置bitwidth
        super(Conv2d, self).__init__(in_places, out_planes, kernel_size, stride, padding, groups, dilation, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth

    def forward(self, x):
        vhat = self.quantize(self.weight, self.bitwidth)
        y = F.conv2d(x, vhat, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        # 这里设置bitwidth
        super(Linear, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth

    def forward(self, x):
        vhat = self.quantize(self.weight, self.bitwidth)
        y = F.linear(x, vhat, self.bias)
        return y
