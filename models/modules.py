import math
import torch.nn as nn
import torch.nn.functional as F
import torch

from timm.models.layers import trunc_normal_, to_2tuple

num_parallel = 2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_2(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class LayerNormProxy(nn.Module):
    """
    Proxy to apply LayerNorm on 2D spatial tensors (B, C, H, W).
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Permute to (B, H, W, C), apply LayerNorm, and permute back
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PyramidCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_cma_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 Convolutions for Projections instead of Linear layers
        self.q_rgb = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv_rgb = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.proj_rgb = nn.Conv2d(dim, dim, 1, 1, 0)

        self.q_depth = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv_depth = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.proj_depth = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial Reduction via strided Convolution and Proxy Norm
        self.sr_cma_ratio = sr_cma_ratio
        if sr_cma_ratio > 1:
            self.sr_rgb = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio),
                LayerNormProxy(dim)
            )
            self.sr_depth = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio),
                LayerNormProxy(dim)
            )

    def forward(self, x, H, W):
        rgb, depth = x[0], x[1]  # (B, N, C)
        B, N, C = rgb.shape
        Nq = H * W

        # Reshape standard sequence (B, N, C) to spatial tensor (B, C, H, W)
        rgb_s = rgb.transpose(1, 2).view(B, C, H, W)
        depth_s = depth.transpose(1, 2).view(B, C, H, W)

        # Generate Queries
        q_rgb = self.q_rgb(rgb_s)
        q_depth = self.q_depth(depth_s)

        # Apply spatial reduction and generate grouped Keys/Values
        if self.sr_cma_ratio > 1:
            rgb_ds = self.sr_rgb(rgb_s)
            kv_rgb = self.kv_rgb(rgb_ds)

            depth_ds = self.sr_depth(depth_s)
            kv_depth = self.kv_depth(depth_ds)
        else:
            kv_rgb = self.kv_rgb(rgb_s)
            kv_depth = self.kv_depth(depth_s)

        # Split grouped KV projections
        k_rgb, v_rgb = torch.chunk(kv_rgb, 2, dim=1)
        k_depth, v_depth = torch.chunk(kv_depth, 2, dim=1)

        Nk = (H // self.sr_cma_ratio) * (W // self.sr_cma_ratio)

        # Reshape and scale for batched matrix multiplication (B * num_heads, head_dim, N)
        q_rgb = q_rgb.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k_rgb = k_rgb.reshape(B * self.num_heads, self.head_dim, Nk)
        v_rgb = v_rgb.reshape(B * self.num_heads, self.head_dim, Nk)

        q_depth = q_depth.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k_depth = k_depth.reshape(B * self.num_heads, self.head_dim, Nk)
        v_depth = v_depth.reshape(B * self.num_heads, self.head_dim, Nk)

        # --- CROSS-MODAL ATTENTION ---
        # RGB queries Depth
        attn_rgb = torch.einsum('b c m, b c n -> b m n', q_rgb, k_depth)
        attn_rgb = F.softmax(attn_rgb, dim=2)
        attn_rgb = self.attn_drop(attn_rgb)
        out_rgb = torch.einsum('b m n, b c n -> b c m', attn_rgb, v_depth)

        # Depth queries RGB
        attn_depth = torch.einsum('b c m, b c n -> b m n', q_depth, k_rgb)
        attn_depth = F.softmax(attn_depth, dim=2)
        attn_depth = self.attn_drop(attn_depth)
        out_depth = torch.einsum('b m n, b c n -> b c m', attn_depth, v_rgb)

        # Reshape back to spatial tensor and apply final projection
        out_rgb = out_rgb.reshape(B, C, H, W)
        out_rgb = self.proj_drop(self.proj_rgb(out_rgb))

        out_depth = out_depth.reshape(B, C, H, W)
        out_depth = self.proj_drop(self.proj_depth(out_depth))

        # Return to Transformer Sequence format (B, N, C)
        out_rgb = out_rgb.flatten(2).transpose(1, 2)
        out_depth = out_depth.flatten(2).transpose(1, 2)

        return [out_rgb, out_depth]


class CVCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Projections
        self.q_rgb = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_rgb = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_rgb = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_depth = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_depth = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_depth = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_depth = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial Reduction (Global Modeling)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_rgb = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_rgb = nn.LayerNorm(dim)
            self.sr_depth = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_depth = nn.LayerNorm(dim)

        # --- THE NOVELTY: Local Geometry Injection via DWConv ---
        self.cv_rgb = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=qkv_bias)
        self.cv_depth = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=qkv_bias)

    def forward(self, rgb, depth, H, W):
        B, N, C = rgb.shape

        # 1. Generate Queries
        q_rgb = self.q_rgb(rgb).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_depth = self.q_depth(depth).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 2. Spatial Reduction for Keys and Values
        if self.sr_ratio > 1:
            rgb_sr = rgb.permute(0, 2, 1).reshape(B, C, H, W)
            rgb_sr = self.sr_rgb(rgb_sr).reshape(B, C, -1).permute(0, 2, 1)
            rgb_sr = self.norm_rgb(rgb_sr)

            depth_sr = depth.permute(0, 2, 1).reshape(B, C, H, W)
            depth_sr = self.sr_depth(depth_sr).reshape(B, C, -1).permute(0, 2, 1)
            depth_sr = self.norm_depth(depth_sr)
        else:
            rgb_sr, depth_sr = rgb, depth

        k_rgb = self.k_rgb(rgb_sr).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_depth = self.k_depth(depth_sr).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 3. Generate Values with Local Feature Mixing (CV-CMA)
        v_rgb_base = self.v_rgb(rgb_sr)
        v_depth_base = self.v_depth(depth_sr)

        h_sr, w_sr = H // self.sr_ratio, W // self.sr_ratio
        v_rgb_spatial = v_rgb_base.permute(0, 2, 1).reshape(B, C, h_sr, w_sr)
        v_depth_spatial = v_depth_base.permute(0, 2, 1).reshape(B, C, h_sr, w_sr)

        # Apply 3x3 Depth-Wise Convolution to bake in local structural geometry
        v_rgb_local = self.cv_rgb(v_rgb_spatial).flatten(2).permute(0, 2, 1)
        v_depth_local = self.cv_depth(v_depth_spatial).flatten(2).permute(0, 2, 1)

        # Combine global values and local values
        v_rgb = (v_rgb_base + v_rgb_local).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_depth = (v_depth_base + v_depth_local).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 4. Cross-Modal Attention Computation
        attn_rgb_depth = (q_rgb @ k_depth.transpose(-2, -1)) * self.scale
        attn_rgb_depth = self.attn_drop(attn_rgb_depth.softmax(dim=-1))
        out_rgb = (attn_rgb_depth @ v_depth).transpose(1, 2).reshape(B, N, C)

        attn_depth_rgb = (q_depth @ k_rgb.transpose(-2, -1)) * self.scale
        attn_depth_rgb = self.attn_drop(attn_depth_rgb.softmax(dim=-1))
        out_depth = (attn_depth_rgb @ v_rgb).transpose(1, 2).reshape(B, N, C)

        out_rgb = self.proj_drop(self.proj_rgb(out_rgb))
        out_depth = self.proj_drop(self.proj_depth(out_depth))

        return out_rgb, out_depth


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_cma_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Projections for cross-modal attention
        self.q_rgb = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_depth = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv_rgb = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_depth = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_depth = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Add spatial reduction ratio support
        self.sr_cma_ratio = sr_cma_ratio
        if sr_cma_ratio > 1:
            self.sr_rgb = nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio)
            self.sr_depth = nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio)
            self.norm_rgb = nn.LayerNorm(dim)
            self.norm_depth = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        rgb_feat, depth_feat = x[0], x[1]  # (B, N, C)
        B, N, C = rgb_feat.shape

        # Query projections - (B, n_heads, N, C//n_heads)
        q_rgb = self.q_rgb(rgb_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_depth = self.q_depth(depth_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Key and Value projections with spatial reduction if needed
        if self.sr_cma_ratio > 1:
            # Process RGB features
            rgb_kv = rgb_feat.permute(0, 2, 1).reshape(B, C, H, W)
            rgb_kv = self.kv_rgb(self.norm_rgb(self.sr_rgb(rgb_kv).reshape(B, C, -1).permute(0, 2, 1)))

            # Process Depth features
            depth_kv = depth_feat.permute(0, 2, 1).reshape(B, C, H, W)
            depth_kv = self.kv_depth(self.norm_depth(self.sr_depth(depth_kv).reshape(B, C, -1).permute(0, 2, 1)))
        else:
            rgb_kv = self.kv_rgb(rgb_feat)
            depth_kv = self.kv_depth(depth_feat)

        # Reshape key-value pairs - (B, n_heads, N_reduced, C//n_heads)
        rgb_kv = rgb_kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_rgb, v_rgb = rgb_kv[0], rgb_kv[1]

        depth_kv = depth_kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_depth, v_depth = depth_kv[0], depth_kv[1]

        # RGB attending to Depth with learned modulation
        attn_rgb_depth = (q_rgb @ k_depth.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_rgb_depth = attn_rgb_depth.softmax(dim=-1)  # (B, num_heads, N, N)
        attn_rgb_depth = self.attn_drop(attn_rgb_depth)  # (B, num_heads, N, N)
        rgb_attended = (attn_rgb_depth @ v_depth).transpose(1, 2).reshape(B, N, C)
        rgb_out = self.proj_drop(self.proj_rgb(rgb_attended))

        # Depth attending to RGB with the same modulation
        attn_depth_rgb = (q_depth @ k_rgb.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_depth_rgb = attn_depth_rgb.softmax(dim=-1)  # (B, num_heads, N, N)
        attn_depth_rgb = self.attn_drop(attn_depth_rgb)  # (B, num_heads, N, N)
        depth_attended = (attn_depth_rgb @ v_rgb).transpose(1, 2).reshape(B, N, C)
        depth_out = self.proj_drop(self.proj_depth(depth_attended))

        return [rgb_out, depth_out]


class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 dilation=[1, 3, 5], groups=1, bias=True,
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes % groups == 0
        assert kernel_size == 3, 'only support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size ** 2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output


class AtrousCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_cma_ratio=1, use_atrous=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_cma_ratio = sr_cma_ratio
        self.use_ds_conv = use_atrous

        # Queries
        self.q_rgb = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_depth = nn.Linear(dim, dim, bias=qkv_bias)

        # Keys and Values
        self.kv_rgb = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_depth = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Dilated depthwise separable convolution
        if self.use_ds_conv:
            self.ds_rgb = ds_conv2d(dim, dim, kernel_size=3, stride=1, groups=dim, bias=qkv_bias)
            self.ds_depth = ds_conv2d(dim, dim, kernel_size=3, stride=1, groups=dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_depth = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_cma_ratio > 1:
            self.sr_rgb = nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio)
            self.sr_depth = nn.Conv2d(dim, dim, kernel_size=sr_cma_ratio, stride=sr_cma_ratio)
            self.norm_rgb = nn.LayerNorm(dim)
            self.norm_depth = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        rgb_feat, depth_feat = x[0], x[1]  # (B, N, C)
        B, N, C = rgb_feat.shape

        # Query Projections
        q_rgb = self.q_rgb(rgb_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_depth = self.q_depth(depth_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.use_ds_conv:
            # Apply dilated depthwise separable convolution
            q_rgb = q_rgb.permute(0, 1, 3, 2).reshape(B, C, H, W)
            q_rgb = self.ds_rgb(q_rgb).reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)

            q_depth = q_depth.permute(0, 1, 3, 2).reshape(B, C, H, W)
            q_depth = self.ds_depth(q_depth).reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)

        # Spatial reduction for KV if needed
        if self.sr_cma_ratio > 1:
            rgb_kv = self.norm_rgb(self.sr_rgb(rgb_feat.permute(0, 2, 1).reshape(B, C, H, W))
                                   .reshape(B, C, -1).permute(0, 2, 1))
            depth_kv = self.norm_depth(self.sr_depth(depth_feat.permute(0, 2, 1).reshape(B, C, H, W))
                                       .reshape(B, C, -1).permute(0, 2, 1))
        else:
            rgb_kv = rgb_feat
            depth_kv = depth_feat

        rgb_kv = self.kv_rgb(rgb_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_rgb, v_rgb = rgb_kv[0], rgb_kv[1]

        depth_kv = self.kv_depth(depth_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_depth, v_depth = depth_kv[0], depth_kv[1]

        # RGB attending to Depth
        attn_rgb = (q_rgb @ k_depth.transpose(-2, -1)) * self.scale
        attn_rgb = self.attn_drop(attn_rgb.softmax(dim=-1))
        rgb_out = (attn_rgb @ v_depth).transpose(1, 2).reshape(B, N, C)
        rgb_out = self.proj_drop(self.proj_rgb(rgb_out))

        # Depth attending to RGB
        attn_depth = (q_depth @ k_rgb.transpose(-2, -1)) * self.scale
        attn_depth = self.attn_drop(attn_depth.softmax(dim=-1))
        depth_out = (attn_depth @ v_rgb).transpose(1, 2).reshape(B, N, C)
        depth_out = self.proj_drop(self.proj_depth(depth_out))

        return [rgb_out, depth_out]


# --- Utility: Window partition & reverse (same as Swin) ---
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# --- Cross-modal attention inside windows ---
class WindowCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.q_rgb = nn.Linear(dim, dim)
        self.q_depth = nn.Linear(dim, dim)
        self.kv_rgb = nn.Linear(dim, dim * 2)
        self.kv_depth = nn.Linear(dim, dim * 2)
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_depth = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, rgb, depth):
        B, N, C = rgb.shape

        q_rgb = self.q_rgb(rgb).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_depth = self.q_depth(depth).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_rgb = self.kv_rgb(rgb).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_depth = self.kv_depth(depth).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_rgb, v_rgb = kv_rgb[0], kv_rgb[1]
        k_depth, v_depth = kv_depth[0], kv_depth[1]

        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        ).permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)

        attn_rgb = (q_rgb @ k_depth.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb + rel_pos_bias
        attn_rgb = attn_rgb.softmax(dim=-1)
        out_rgb = (attn_rgb @ v_depth).transpose(1, 2).reshape(B, N, C)
        out_rgb = self.proj_rgb(out_rgb)

        attn_depth = (q_depth @ k_rgb.transpose(-2, -1)) * self.scale
        attn_depth = attn_depth + rel_pos_bias
        attn_depth = attn_depth.softmax(dim=-1)
        out_depth = (attn_depth @ v_rgb).transpose(1, 2).reshape(B, N, C)
        out_depth = self.proj_depth(out_depth)

        return out_rgb, out_depth

# --- SWCMA Block ---
class SWCMABlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)
        self.cross_attn = WindowCrossModalAttention(dim, num_heads)

    def forward(self, rgb_org, depth_org, H, W):
        B, N, C = rgb_org.shape
        rgb = self.norm1(rgb_org).view(B, H, W, C)
        depth = self.norm1(depth_org).view(B, H, W, C)

        # Shift if needed
        if self.shift_size > 0:
            rgb = torch.roll(rgb, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            depth = torch.roll(depth, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Pad height and width to be divisible by window size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size

        rgb = F.pad(rgb, (0, 0, 0, pad_r, 0, pad_b))  # pad right and bottom
        depth = F.pad(depth, (0, 0, 0, pad_r, 0, pad_b))

        Hp, Wp = rgb.shape[1], rgb.shape[2]

        # Partition windows
        rgb_windows = window_partition(rgb, self.window_size).view(-1, self.window_size**2, C)
        depth_windows = window_partition(depth, self.window_size).view(-1, self.window_size**2, C)

        # Apply window cross-modal attention
        out_rgb, out_depth = self.cross_attn(rgb_windows, depth_windows)

        # Reverse windows
        out_rgb = out_rgb.view(-1, self.window_size, self.window_size, C)
        out_rgb = window_reverse(out_rgb, self.window_size, Hp, Wp)
        out_depth = out_depth.view(-1, self.window_size, self.window_size, C)
        out_depth = window_reverse(out_depth, self.window_size, Hp, Wp)

        out_rgb = out_rgb[:, :H, :W, :].contiguous()
        out_depth = out_depth[:, :H, :W, :].contiguous()

        # Reverse shift
        if self.shift_size > 0:
            out_rgb = torch.roll(out_rgb, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            out_depth = torch.roll(out_depth, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        rgb_attn = rgb_org + out_rgb.view(B, H * W, C)
        depth_attn = depth_org + out_depth.view(B, H * W, C)

        # mlp
        rgb_out = rgb_attn + self.mlp(self.norm2(rgb_attn), H, W)
        depth_out = depth_attn + self.mlp(self.norm2(depth_attn), H, W)

        return rgb_out, depth_out

# --- SWCMA Layer ---
class SWCMALayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            SWCMABlock(dim, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        rgb, depth = x[0], x[1]

        for blk in self.blocks:
            rgb, depth = blk(rgb, depth, H, W)
        return rgb, depth


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]