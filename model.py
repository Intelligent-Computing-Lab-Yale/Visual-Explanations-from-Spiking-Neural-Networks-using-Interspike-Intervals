"""
    Overall code flow is from Rathi et al. (2020) [https://github.com/nitin-rathi/hybrid-snn-conversion]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.autograd import Variable



# --------------------------------------------------
# Spiking neuron with piecewise-linear surrogate gradient
# --------------------------------------------------
class LinearSpike(torch.autograd.Function):
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * LinearSpike.gamma * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


# --------------------------------------------------
# Spiking neuron with pass-through surrogate gradient
# --------------------------------------------------
class PassThruSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# Overwrite the naive spike function by differentiable spiking nonlinearity which implements a surrogate gradient
def init_spike_fn(grad_type):
    if (grad_type == 'Linear'):
        spike_fn = LinearSpike.apply
    elif (grad_type == 'PassThru'):
        spike_fn = PassThruSpike.apply
    else:
        sys.exit("Unknown gradient type '{}'".format(grad_type))
    return spike_fn


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class SNN_VGG11(nn.Module):
    def __init__(self, num_timestep=30, leak_mem=0.99):
        super(SNN_VGG11, self).__init__()

        self.img_size = 64
        self.num_steps = num_timestep
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        affine_flag = True
        bias_flag = False

        # Instantiate the ConvSNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn2_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn3_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn4_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn5_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn6_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn7_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn8_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AvgPool2d(kernel_size=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bnfc_list = nn.ModuleList(
            [nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
             range(self.batch_num)])
        self.fc2 = nn.Linear(4096, 200, bias=bias_flag)

        batchnormlist = [self.bn1_list, self.bn2_list, self.bn3_list, self.bn4_list, self.bn5_list,
                         self.bn6_list, self.bn7_list, self.bn8_list, self.bnfc_list]

        for bnlist in batchnormlist:
            for bnbn in bnlist:
                bnbn.bias = None

            # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0

        self.saved_forward= []

        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn('Linear')
        self.spike_pool = init_spike_fn('PassThru')


    def forward(self, inp, target_layer=2):
        outputList = []

        batch_size = inp.size(0)
        h, w = inp.size(2) ,inp.size(3)

        mem_conv1 = Variable(torch.zeros(batch_size, 64, h, w), requires_grad=True).cuda()
        mem_conv2 = Variable(torch.zeros(batch_size, 128, h//2, w//2).cuda(), requires_grad=True)
        mem_conv3 = Variable(torch.zeros(batch_size, 256, h//4, w//4).cuda(), requires_grad=True)
        mem_conv4 = Variable(torch.zeros(batch_size, 256, h//4, w//4).cuda(), requires_grad=True)
        mem_conv5 = Variable(torch.zeros(batch_size, 512, h//8, w//8).cuda(), requires_grad=True)
        mem_conv6 = Variable(torch.zeros(batch_size, 512, h//8, w//8).cuda(), requires_grad=True)
        mem_conv7 = Variable(torch.zeros(batch_size, 512, h // 16, w// 16).cuda(), requires_grad=True)
        mem_conv8 = Variable(torch.zeros(batch_size, 512, h// 16, w// 16).cuda(), requires_grad=True)

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, 200).cuda()

        for t in range(self.num_steps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            # Compute the conv1 outputs
            mem_thr   = (mem_conv1/self.conv1.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr>0] = self.conv1.threshold
            mem_conv1 = (self.leak_mem*mem_conv1 + self.bn1_list[int(t)](self.conv1(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool1 outputs
            out =  self.pool1(out_prev)
            out_prev = out.clone()

            # Compute the conv2 outputs
            mem_thr   = (mem_conv2/self.conv2.threshold) - 1.0
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr>0] = self.conv2.threshold
            mem_conv2 = (self.leak_mem*mem_conv2 + self.bn2_list[int(t)](self.conv2(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool2 outputs
            out = self.pool2(out_prev)
            out_prev = out.clone()

            # Compute the conv3 outputs
            mem_thr = (mem_conv3 / self.conv3.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = self.conv3.threshold
            mem_conv3 = (self.leak_mem * mem_conv3 + self.bn3_list[int(t)](self.conv3(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv4 outputs
            mem_thr = (mem_conv4 / self.conv4.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = self.conv4.threshold
            mem_conv4 = (self.leak_mem * mem_conv4 + self.bn4_list[int(t)](self.conv4(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 4:
                self.saved_forward.append(out_prev)

            # Compute the avgpool3 outputs
            out = self.pool3(out_prev)
            out_prev = out.clone()

            # Compute the conv5 outputs
            mem_thr = (mem_conv5 / self.conv5.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = self.conv5.threshold
            mem_conv5 = (self.leak_mem * mem_conv5 + self.bn5_list[int(t)](self.conv5(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv6 outputs
            mem_thr = (mem_conv6 / self.conv6.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.conv6.threshold
            mem_conv6 = (self.leak_mem * mem_conv6 + self.bn6_list[int(t)](self.conv6(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 6:
                self.saved_forward.append(out_prev)


            # Compute the avgpool4 outputs
            out = self.pool4(out_prev)
            out_prev = out.clone()

            # Compute the conv7 outputs
            mem_thr = (mem_conv7 / self.conv7.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = self.conv7.threshold
            mem_conv7 = (self.leak_mem * mem_conv7 + self.bn7_list[int(t)](self.conv7(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv8 outputs
            mem_thr = (mem_conv8 / self.conv8.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv8).cuda()
            rst[mem_thr > 0] = self.conv8.threshold
            mem_conv8 = (self.leak_mem * mem_conv8 + self.bn8_list[int(t)](self.conv8(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 8:
                self.saved_forward.append(out_prev)

            # Compute the avgpool5 outputs
            out = self.avg_pool(out_prev)
            out_prev = out.clone()
            out_prev = out_prev.reshape(batch_size, -1)

            # compute fc1
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = (self.leak_mem * mem_fc1 + self.bnfc_list[int(t)](self.fc1(out_prev)) - rst)

            out_prev = out.clone()
            mem_fc2 = (1 * mem_fc2 + self.fc2(out_prev))
            out_voltage_tmp = (mem_fc2) / (t+1e-3)
            outputList.append(out_voltage_tmp)

        out_voltage  = mem_fc2
        out_voltage = (out_voltage) / self.num_steps

        return out_voltage


