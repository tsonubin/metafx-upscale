"""Minimal PyTorch architectures for the benchmark candidates, vendored so we don't
need basicsr / mmcv (both break on current torch and have CUDA-only extras).
Layer names match the official checkpoints so they load with strict=True."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- Real-ESRGAN
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """RealESRGAN_x4plus generator (ESRGAN RRDB, 23 blocks, 16.7M params)."""

    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """realesr-general-x4v3 (plain VGG-style, 1.2M params)."""

    def __init__(self, nf=64, nc=32, scale=4):
        super().__init__()
        self.scale = scale
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)]
        for _ in range(nc):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        body.append(nn.Conv2d(nf, 3 * scale * scale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


# ---------------------------------------------------------------- RealBasicVSR
def flow_warp(x, flow, padding_mode="zeros"):
    """Warp x (n,c,h,w) by flow (n,h,w,2) in pixels, mmedit convention."""
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((gx, gy), 2) + flow
    gx = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gx, gy), 3), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


class ConvModule(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.act else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModule(8, 32), ConvModule(32, 64), ConvModule(64, 32), ConvModule(32, 16),
            ConvModule(16, 2, act=False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = flow if level == 0 else F.interpolate(
                flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), "border"), flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, cin, nf=64, nb=30):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                  nn.Sequential(*[ResidualBlockNoBN(nf) for _ in range(nb)]))

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, cin, cout, scale):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(cin, cout * scale * scale, 3, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class BasicVSRNet(nn.Module):
    """Bidirectional recurrent VSR: SPyNet flow, flow-warped feature propagation both ways."""

    def __init__(self, nf=64, nb=20):
        super().__init__()
        self.nf = nf
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(nf + 3, nf, nb)
        self.forward_resblocks = ResidualBlocksWithInputConv(nf + 3, nf, nb)
        self.fusion = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.upsample1 = PixelShufflePack(nf, nf, 2)
        self.upsample2 = PixelShufflePack(nf, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, True)

    def forward(self, lrs, on_frame):
        """lrs (1,t,3,h,w) in [0,1]; calls on_frame(i, hr) for each output instead of stacking."""
        n, t, c, h, w = lrs.shape
        a = lrs[:, :-1].reshape(-1, c, h, w)
        b = lrs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(a, b).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(b, a).view(n, t - 1, 2, h, w)

        backward = [None] * t
        feat = lrs.new_zeros(n, self.nf, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat = flow_warp(feat, flows_backward[:, i].permute(0, 2, 3, 1))
            feat = self.backward_resblocks(torch.cat([lrs[:, i], feat], 1))
            backward[i] = feat

        feat = torch.zeros_like(feat)
        for i in range(t):
            lr = lrs[:, i]
            if i > 0:
                feat = flow_warp(feat, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            feat = self.forward_resblocks(torch.cat([lr, feat], 1))
            out = self.lrelu(self.fusion(torch.cat([backward[i], feat], 1)))
            backward[i] = None
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.conv_last(self.lrelu(self.conv_hr(out)))
            out = out + F.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False)
            on_frame(i, out)


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR: per-frame image-cleaning stage, then BasicVSR. One cleaning pass
    (the official config's dynamic_refine_thres=255 always stops after the first)."""

    def __init__(self, nf=64, n_prop=20, n_clean=20):
        super().__init__()
        self.image_cleaning = nn.Sequential(ResidualBlocksWithInputConv(3, nf, n_clean),
                                            nn.Conv2d(nf, 3, 3, 1, 1))
        self.basicvsr = BasicVSRNet(nf, n_prop)

    def forward(self, lqs, on_frame):
        n, t, c, h, w = lqs.shape
        cleaned = []
        for i in range(t):  # frame by frame to bound memory
            x = lqs[:, i]
            cleaned.append(x + self.image_cleaning(x))
        self.basicvsr(torch.stack(cleaned, 1), on_frame)
