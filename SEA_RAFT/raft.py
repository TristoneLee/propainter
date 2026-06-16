import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from .update import BasicUpdateBlock
from .corr import CorrBlock
from .utils.utils import coords_grid, InputPadder
from .extractor import ResNetFPN
from .layer import conv1x1, conv3x3

from huggingface_hub import PyTorchModelHubMixin

class RAFT(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="https://github.com/princeton-vl/SEA-RAFT",
    pipeline_tag="optical-flow-estimation",
    license="bsd-3-clause",
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.size == 'small':
            args.use_var = True
            args.var_min = 0
            args.var_max = 10
            args.pretrain = 'resnet18'
            args.initial_dim = 64
            args.block_dims = [64, 128, 256]
            args.radius = 4
            args.dim = 128
            args.num_blocks = 2
            args.iters = 0
        elif args.size == 'medium':
            args.use_var = True
            args.var_min = 0
            args.var_max = 10
            args.pretrain = 'resnet34'
            args.initial_dim = 64
            args.block_dims = [64, 128, 256]
            args.radius = 4
            args.dim = 128
            args.num_blocks = 2
            args.iters = 4
        self.output_dim = args.dim * 2
        
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        # if args.iters > 0:
        self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
        self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)
        if args.size == 'small':
            self.load_state_dict(torch.load('weights/Tartan-C-T-TSKH432x960-S.pth', map_location='cpu'))
        elif args.size == 'medium':
            self.load_state_dict(torch.load('weights/Tartan-C-T-TSKH432x960-M.pth', map_location='cpu'))
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)
        coords2 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)
        return coords1, coords2

    def upsample_data(self, flow, mask):
        """ Upsample [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward_bi(self, image1, image2, iters=None, test_mode=False):
        """Compute forward and backward flow in a single forward pass."""
        img1 = torch.cat([image1, image2], dim=0)
        img2 = torch.cat([image2, image1], dim=0)
        _, flow = self.forward(img1, img2, iters=iters, test_mode=test_mode)
        n = image1.shape[0]
        return flow[:n], flow[n:]

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        N, _, H, W = image1.shape
        iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)


        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device, dtype=image1.dtype)

        with record_function("cnet"):
            cnet = self.cnet(torch.cat([image1, image2], dim=1))
            cnet = self.init_conv(cnet)
            net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        with record_function("init_flow"):
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_update[:, :2]
            flow_up = self.upsample_data(flow_8x, weight_update)
            flow_predictions.append(flow_up)

        if self.args.iters > 0:
            with record_function("fnet"):
                fmap1_8x = self.fnet(image1)
                fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            with record_function("corr"):
                coords2 = (coords_grid(N, H, W, device=image1.device, dtype=image1.dtype) + flow_8x).detach()
                corr = corr_fn(coords2, dilation=dilation)
            with record_function("update_block"):
                net = self.update_block(net, context, corr, flow_8x)
            with record_function("upsample"):
                flow_update = self.flow_head(net)
                weight_update = .25 * self.upsample_weight(net)
                flow_8x = flow_8x + flow_update[:, :2]
                flow_up = self.upsample_data(flow_8x, weight_update)
            flow_predictions.append(flow_up)

        for i in range(len(flow_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])

        return None, flow_predictions[-1]