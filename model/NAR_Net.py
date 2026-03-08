import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math
import os
import numpy as np
from typing import cast, List

# ====================================================================================
# SECTION 1: 基础组件 (Basic Components)
# ====================================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() != 4:
             return self.body(x)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners,
                          recompute_scale_factor=True)
        return out

class AdaConv(nn.Module):
    """
    自适应动态卷积 (Adaptive Convolution) — 基于基卷积分解的高效实现。

    Conv(x, sum(w_i * a_i)) == sum(Conv(x, w_i) * a_i) 用于移除 unfold 与巨型动态权重。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bases=4, bias=False):
        super(AdaConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.bases = bases

        # 保持原始参数形状以兼容旧 checkpoint。
        self.weight = nn.Parameter(torch.empty((1, groups, (out_channels // groups), (in_channels // groups) * kernel_size ** 2, bases)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty((1, groups, out_channels // groups, bases)))
            fan_in = (in_channels // groups) * kernel_size ** 2
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, bases, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(bases, bases, 3, 1, 1)
        )

    def dynamic_conv(self, x, para):
        bs, _, h, w = x.shape
        g = self.groups
        out_g = self.out_channels // g
        in_g = self.in_channels // g
        k = self.kernel_size
        bases = self.bases

        # 将存储的多基底权重量化为标准 group conv 权重，以复用 cuDNN。
        weight_tensor = self.weight.squeeze(0).view(g, out_g, in_g, k, k, bases)
        weight_tensor = weight_tensor.permute(0, 1, 5, 2, 3, 4).contiguous()
        w_conv = weight_tensor.view(g * out_g * bases, in_g, k, k)

        conv_out = F.conv2d(x, w_conv, stride=self.stride, padding=self.padding, groups=g)
        conv_out = conv_out.view(bs, g, out_g, bases, h, w)

        para_expanded = para.view(bs, 1, 1, bases, h, w)
        out = (conv_out * para_expanded).sum(dim=3).reshape(bs, self.out_channels, h, w)

        if self.bias is not None:
            b = self.bias.view(1, g, out_g, bases, 1, 1)
            b_dyn = (b * para_expanded).sum(dim=3)
            out = out + b_dyn.reshape(bs, self.out_channels, h, w)

        return out

    def forward(self, x):
        para = self.predictor(x)
        return self.dynamic_conv(x, para)

# ====================================================================================
# SECTION 2: 核心模块 (DLM, FFN, ReGroup, ESA)
# ====================================================================================

class CREmbedding(nn.Module):
    """将标量 CR 映射为特征向量"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(), # 平滑激活函数
            nn.Linear(dim, dim),
        )
    
    def forward(self, cr):
        # 确保输入是 (B, 1) 的 Tensor
        if not torch.is_tensor(cr):
            device = self.net[0].weight.device
            dtype = self.net[0].weight.dtype
            cr = torch.tensor([cr], device=device, dtype=dtype)
        elif cr.dtype != self.net[0].weight.dtype:
             cr = cr.to(dtype=self.net[0].weight.dtype)
             
        if cr.dim() == 0: cr = cr.view(1, 1)
        elif cr.dim() == 1: cr = cr.unsqueeze(1)
        return self.net(cr)

class DLM(nn.Module):
    """Dynamic Local Modulation (DLM) — 通过自适应基卷积分解实现的局部纹理建模模块。"""
    def __init__(self, dim, out_dim=None, scale=1.0, bias=True, ks=3):
        super(DLM, self).__init__()
        if dim == 0:
             self.identity = nn.Identity()
             return
        if out_dim is None: out_dim = dim
        mid_dim = int(dim * scale)
        
        self.conv_in = nn.Conv2d(dim, mid_dim * 2, 1, 1, 0, bias=bias)
        self.dw_static = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, bias=bias)
        self.dw_adaptive = nn.Sequential(
            AdaConv(mid_dim, mid_dim, kernel_size=ks, padding=ks//2, groups=mid_dim, bases=4, bias=False),
            nn.GELU()
        )
        self.conv_out = nn.Conv2d(mid_dim, out_dim, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        if hasattr(self, 'identity'): return self.identity(x)
        x_dual = self.conv_in(x)
        x_static, x_adapt = x_dual.chunk(2, dim=1)
        x_static = self.dw_static(x_static)
        x_adapt = self.dw_adaptive(x_adapt)
        x_fused = x_static * x_adapt
        out = self.conv_out(x_fused)
        return out

class GatedFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super(GatedFeedForward, self).__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class ReGroup(nn.Module):
    def __init__(self, groups_ratio=[1, 2, 2, 3]):
        super(ReGroup, self).__init__()
        self.groups_ratio = groups_ratio

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        
        with torch.no_grad():
            channel_features = q.reshape(B, C, -1)
            mean = channel_features.mean(dim=2, keepdim=True)
            centered = channel_features - mean
            cov = torch.bmm(centered, centered.transpose(1, 2))
            
            var = torch.diagonal(cov, dim1=1, dim2=2)
            std = torch.sqrt(var + 1e-8).unsqueeze(2)
            denominator = torch.bmm(std, std.transpose(1, 2)).clamp(min=1e-8)
            correlation_matrix = cov / denominator
            
            mean_similarity = correlation_matrix.mean(dim=2)
            _, sorted_indices = torch.sort(mean_similarity, descending=True, dim=1)

        idx_expanded = sorted_indices.view(B, C, 1, 1).expand(B, C, H, W)
        q_sorted = torch.gather(q, 1, idx_expanded)
        k_sorted = torch.gather(k, 1, idx_expanded)
        v_sorted = torch.gather(v, 1, idx_expanded)

        total_ratio = sum(self.groups_ratio)
        group_dims = [int(r / total_ratio * C) for r in self.groups_ratio]
        group_dims[-1] = C - sum(group_dims[:-1])
        
        if any(d <= 0 for d in group_dims):
            group_dims = [C // len(group_dims)] * len(group_dims)
            group_dims[-1] = C - sum(group_dims[:-1])

        q_groups = torch.split(q_sorted, group_dims, dim=1)
        k_groups = torch.split(k_sorted, group_dims, dim=1)
        v_groups = torch.split(v_sorted, group_dims, dim=1)

        return (q_groups, k_groups, v_groups), sorted_indices

    def restore(self, x, indices):
        B, C, H, W = x.shape
        inverse_indices = torch.argsort(indices, dim=1)
        idx_expanded = inverse_indices.view(B, C, 1, 1).expand(B, C, H, W)
        return torch.gather(x, 1, idx_expanded)

class Intra_CacheModulation(nn.Module):
    def __init__(self, dim):
        super(Intra_CacheModulation, self).__init__()
        self.down = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.up = nn.Conv2d(dim // 2, dim, kernel_size=1)
        self.gatingConv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, features, cache):
        mixed = features + cache
        x_gated = F.gelu(self.gatingConv(mixed)) * mixed
        mod = self.up(self.down(x_gated))
        return features + mod

class ESA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=1, num_heads=4, bias=False, ks=3, sr=2):
        super(ESA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.sr = sr
        self.ch = ch
        self.num_heads = num_heads

        # [Optimized] 统一通道比例为均匀分布，以便进行并行张量运算
        # 注意：这要求 ch 必须能被 num_heads 整除，否则并行 stack 会失败
        self.ratios = [1] * num_heads
        
        self.head_dim = ch // num_heads
        # 简单兼容处理：ReGroup 仍然使用 group_dims 逻辑，这里确保它是均匀的
        self.group_dims = [self.head_dim] * num_heads
        if ch % num_heads != 0:
             self.group_dims[-1] += ch % num_heads

        if sr > 1 and ch > 0:
            self.sampler = nn.MaxPool2d(kernel_size=sr, stride=sr)
            self.LocalProp = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=ks, stride=1, padding=ks//2, groups=ch, bias=True, padding_mode='reflect'),
                Interpolate(scale_factor=sr, mode='bilinear', align_corners=True),
            )

        if ch > 0:
            self.qkv_conv = nn.Conv2d(ch, ch*3, kernel_size=1, bias=bias)
            self.lce = nn.Conv2d(ch*3, ch*3, kernel_size=3, stride=1, padding=1, groups=ch*3, bias=bias)

            self.regroup = ReGroup(self.ratios)
            self.intra_mod = Intra_CacheModulation(ch)

            # [Optimized] 使用分组卷积实现并行的门控 (Gating)
            # 等价于对每个 Head 独立应用 Linear，但在空间维度上执行效率更高
            self.gate_conv = nn.Conv2d(ch, ch, kernel_size=1, groups=num_heads)
            nn.init.constant_(self.gate_conv.bias, 1.0) # type: ignore
            nn.init.xavier_normal_(self.gate_conv.weight, gain=0.1)

            # [Optimized] 并行化的 Temperature 参数 (1, Heads, 1, 1)
            self.temp = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

            # [Optimized] 并行化的位置编码 (支持广播)
            neighborhood_size = block_size + 2 * halo_size
            self.neighborhood_kernel = neighborhood_size
            
            h_d = self.head_dim // 2
            w_d = self.head_dim - h_d
            
            # 使用独立参数以赋予每个 Head 不同的空间偏置，弥补统一通道带来的多样性损失
            self.rel_h_emb = nn.Parameter(torch.randn(1, num_heads, neighborhood_size, 1, h_d) * 0.02)
            self.rel_w_emb = nn.Parameter(torch.randn(1, num_heads, 1, neighborhood_size, w_d) * 0.02)
            
            self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)

    def forward(self, x):
        if self.ch == 0: return x
        x_in = self.sampler(x) if self.sr > 1 else x
        
        B, C, H, W = x_in.size()
        pad_r = (self.block_size - W % self.block_size) % self.block_size
        pad_b = (self.block_size - H % self.block_size) % self.block_size
        x_padded = F.pad(x_in, (0, pad_r, 0, pad_b), mode='reflect') if pad_r > 0 or pad_b > 0 else x_in

        qkv = self.lce(self.qkv_conv(x_padded))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        (q_groups, k_groups, v_groups), sorted_indices = self.regroup(q, k, v)
        
        # [Vectorized Operation Start]
        # 1. Stack tensors: (B, NumHeads, HeadDim, H, W)
        q_stack = torch.stack(q_groups, dim=1)
        k_stack = torch.stack(k_groups, dim=1)
        v_stack = torch.stack(v_groups, dim=1)
        
        # 2. Compute Gating Score Efficiently on Spatial Features
        # View as (B, Channels, H, W) for Conv2d. Grouped Conv maps each head's channels to itself.
        gate_in = q_stack.view(B, C, H, W) 
        gate_score = self.gate_conv(gate_in)
        # Reshape back to parallel format for broadcasting later: (B, Heads, Dim, H, W)
        gate_score = gate_score.view(B, self.num_heads, self.head_dim, H, W)

        # 3. Prepare Unfold
        block, halo = self.block_size, self.halo_size
        neighborhood = self.neighborhood_kernel
        h_blocks = x_padded.shape[2] // block
        w_blocks = x_padded.shape[3] // block

        # Query Unfold (Non-overlapping blocks)
        # Transform: (B, Heads, C, H, W) -> (B*Blocks, Heads, BlockSq, C)
        # Note: We keep Heads dimension separate for broadcasting
        qi_unf = rearrange(q_stack, 'b heads c (h k1) (w k2) -> (b h w) heads (k1 k2) c', 
                           k1=block, k2=block)

        # Key/Value Unfold (Overlapping neighborhood)
        # Trick: Merge Heads into Channels to use optimized F.unfold
        k_merged = k_stack.view(B, C, H, W)
        v_merged = v_stack.view(B, C, H, W)
        
        k_unf_raw = F.unfold(k_merged, kernel_size=neighborhood, stride=block, padding=halo)
        v_unf_raw = F.unfold(v_merged, kernel_size=neighborhood, stride=block, padding=halo)
        
        # Reshape Unfold Output: (B, Heads*Dim*K*K, L_blocks) -> (B*L, Heads, K*K, Dim)
        L = k_unf_raw.shape[-1]
        def restore_unfold(t):
            # View: B, Heads, Dim, K, K, L
            t = t.view(B, self.num_heads, self.head_dim, neighborhood, neighborhood, L)
            return rearrange(t, 'b heads c k1 k2 l -> (b l) heads (k1 k2) c')
            
        ki_unf = restore_unfold(k_unf_raw)
        vi_unf = restore_unfold(v_unf_raw)

        # 4. Positional Embeddings (Parallelized)
        h_d = self.head_dim // 2
        w_d = self.head_dim - h_d
        
        # Recover spatial dims of K for embedding addition
        ki_spatial = rearrange(ki_unf, 'bl heads (k1 k2) c -> bl heads k1 k2 c', k1=neighborhood)
        k_h, k_w = torch.split(ki_spatial, [h_d, w_d], dim=-1)
        
        # Broadcasting: (BL, Heads, K, K, h_d) + (1, Heads, K, 1, h_d)
        k_h = k_h + self.rel_h_emb
        k_w = k_w + self.rel_w_emb
        
        ki_unf = rearrange(torch.cat([k_h, k_w], dim=-1), 'bl heads k1 k2 c -> bl heads (k1 k2) c')

        # 5. Attention Computation (Fully Parallel)
        qi_norm = F.normalize(qi_unf, p=2, dim=-1)
        ki_norm = F.normalize(ki_unf, p=2, dim=-1)

        # MatMul: (BL, Heads, N_q, D) @ (BL, Heads, N_k, D)^T -> (BL, Heads, N_q, N_k)
        attn = torch.matmul(qi_norm, ki_norm.transpose(-1, -2))
        attn = attn * torch.exp(self.temp)
        attn = attn.softmax(dim=-1)

        # Aggregation: (BL, Heads, N_q, N_k) @ (BL, Heads, N_k, D) -> (BL, Heads, N_q, D)
        out_i = torch.matmul(attn, vi_unf)

        # 6. Reassemble and Apply Gate
        # Fold back to spatial: (BL, Heads, BlockSq, Dim) -> (B, Heads, Dim, H, W)
        out_folded = rearrange(out_i, '(b h w) heads (k1 k2) c -> b heads c (h k1) (w k2)',
                               b=B, h=h_blocks, w=w_blocks, k1=block, k2=block)
        
        # Apply the pre-computed spatial gate
        out = out_folded * torch.sigmoid(gate_score)
        
        # Flatten heads to channels
        out = out.reshape(B, C, H, W)
        
        # 7. Post-processing
        # Efficient cache concat
        cache_cat = q_stack + k_stack
        cache_cat = cache_cat.reshape(B, C, H, W)

        # Restore channel order
        out = self.regroup.restore(out, sorted_indices)
        cache_cat = self.regroup.restore(cache_cat, sorted_indices)
        
        out = self.intra_mod(out, cache_cat)
        out = self.proj_out(out)

        if self.sr > 1:
            out = self.LocalProp(out)
        if out.size(2) != H or out.size(3) != W:
            out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        elif pad_r > 0 or pad_b > 0:
            out = out[:, :, :x.size(2), :x.size(3)]
        return out

class ChannelCrossAttention(nn.Module):
    def __init__(self, dim, head_dim, bias=False):
        super().__init__()
        if head_dim > 0 and dim > 0:
             self.num_head = math.gcd(dim, head_dim) or 1 if dim % head_dim != 0 else dim // head_dim
        else:
             self.num_head = 1
             if dim == 0:
                  self.identity = nn.Identity()
                  return

        self.temperature = nn.Parameter(torch.ones(self.num_head, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        if hasattr(self, 'identity'): return self.identity(x)
        if y is None: return torch.zeros_like(x)
        if x.shape != y.shape: return torch.zeros_like(x)

        h, w = x.shape[-2::]
        q = self.q_dwconv(self.q(x))
        k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out

class HSB(nn.Module):
    def __init__(self, dim, num_heads=4, bs=8, ks=3, sr=2, scale=2.0, emb_dim=None):
        super(HSB, self).__init__()
        self.dim_mem = dim // 2
        self.dim_esa = dim - self.dim_mem
        self.ln1 = LayerNorm(dim=dim)

        if self.dim_mem > 0:
            self.dlm = DLM(dim=self.dim_mem, out_dim=self.dim_mem, scale=scale, bias=True, ks=ks)
        if self.dim_esa > 0:
            self.esa = ESA(ch=self.dim_esa, block_size=bs, halo_size=1, num_heads=num_heads, bias=False, ks=ks, sr=sr)

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.ln2 = LayerNorm(dim=dim)
        self.ffn = GatedFeedForward(dim=dim, expansion_factor=scale, bias=True)

        # 条件投影层 (生成 Scale 和 Shift)
        cond_in_dim = emb_dim if emb_dim is not None else dim
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_in_dim, dim * 2)
        )
        # 初始化为0，确保训练初期网络行为与无注入时一致 (Zero-Initialization 策略)
        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

    def forward(self, x, cond_emb=None):
        res = self.ln1(x)
        
        # AdaLN 调制逻辑
        if cond_emb is not None:
            # cond_emb: (B, dim) -> style: (B, 2*dim, 1, 1)
            style = self.cond_proj(cond_emb).unsqueeze(2).unsqueeze(3)
            scale, shift = style.chunk(2, dim=1)
            # 仿射变换：放大或平移特征分布
            res = res * (1 + scale) + shift

        split_dims = [d for d in [self.dim_mem, self.dim_esa] if d > 0]
        features = torch.split(res, split_dims, dim=1)

        fused_parts = []
        feat_idx = 0
        if self.dim_mem > 0:
            fused_parts.append(self.dlm(features[feat_idx]))
            feat_idx += 1
        if self.dim_esa > 0:
            fused_parts.append(self.esa(features[feat_idx]))

        fused = torch.cat(fused_parts, dim=1)
        x = self.fuse(fused) + x
        x = self.ffn(self.ln2(x)) + x
        return x

class NAR_Prior(nn.Module):
    def __init__(self, color_channel=1, dim=64, head_dim=32, window_size=8,
                 enc_blocks=[2,2,2], mid_blocks=2, dec_blocks=[2,2,2],
                 hsb_sr=2, hsb_scale=2.0, hsb_ratio=1.0): 
        super(NAR_Prior, self).__init__()
        self.pad_size = window_size * (2 ** len(enc_blocks))
        self.padder = nn.ReplicationPad2d
        self.embedding = nn.Conv2d(color_channel, dim, 3, 1, 1)
        self.mapping = nn.Conv2d(dim, color_channel, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.memorizers = nn.ModuleList()

        current_dim = dim
        effective_head_dim = max(1, head_dim)
        target_heads = max(1, dim // effective_head_dim)

        for num in enc_blocks:
            self.memorizers.append(ChannelCrossAttention(current_dim, head_dim))
            self.encoders.append(nn.Sequential(*[
                HSB(current_dim, num_heads=target_heads, bs=window_size, sr=hsb_sr, scale=hsb_scale, emb_dim=dim)
                for i in range(num)
            ]))
            self.downs.append(nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(current_dim * 4, 2 * current_dim, kernel_size=1, bias=False) 
            ))
            current_dim = current_dim * 2
            target_heads = max(1, current_dim // effective_head_dim)

        self.memorizers.append(ChannelCrossAttention(current_dim, head_dim))
        self.bottleneck = nn.Sequential(*[
            HSB(current_dim, num_heads=target_heads, bs=window_size, sr=hsb_sr, scale=hsb_scale, emb_dim=dim)
            for i in range(mid_blocks)
        ])

        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(current_dim, (current_dim//2) * 4, kernel_size=1, bias=False), 
                nn.PixelShuffle(2)
            ))
            current_dim = current_dim // 2
            target_heads = max(1, current_dim // effective_head_dim)
            self.decoders.append(nn.Sequential(*[
                HSB(current_dim, num_heads=target_heads, bs=window_size, sr=hsb_sr, scale=hsb_scale, emb_dim=dim)
                for i in range(num)
            ]))

    def forward(self, inp, memory=None, beta_k=None, return_features=False, cr_emb=None):
        x = inp
        _, _, H, W = x.shape
        paddingBottom = (self.pad_size - H % self.pad_size) % self.pad_size
        paddingRight = (self.pad_size - W % self.pad_size) % self.pad_size

        if paddingBottom > 0 or paddingRight > 0:
            x = self.padder((0, paddingRight, 0, paddingBottom))(x)
        
        x = self.embedding(x)
        encoder_list = []
        memory_list = [] 
        feature_list: List[torch.Tensor] = []
        memory_reversed = memory[::-1] if memory is not None else None

        # 定义一个辅助函数来处理 Sequential 的显式循环
        def run_stage(stage_blocks, feat, cond):
            for block in stage_blocks:
                if isinstance(block, HSB):
                    feat = block(feat, cond_emb=cond)
                else:
                    feat = block(feat)
            return feat
            
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            memorizer = self.memorizers[i]
            skip_m = memory_reversed[i] if memory_reversed is not None and i < len(memory_reversed) else None
            x = x + memorizer(x, skip_m)
            x = run_stage(encoder, x, cr_emb)
            encoder_list.append(x)
            if return_features: feature_list.append(x)
            x = down(x)

        skip_m = memory[0] if memory is not None and len(memory) > 0 else None
        x = x + self.memorizers[-1](x, skip_m)
        x = run_stage(self.bottleneck, x, cr_emb)
        memory_list.append(x)
        if return_features: feature_list.append(x)

        for decoder, up, skip_x in zip(self.decoders, self.ups, encoder_list[::-1]):
            x = up(x)
            x = x + skip_x
            x = run_stage(decoder, x, cr_emb)
            memory_list.append(x)
            if return_features: feature_list.append(x)

        x = self.mapping(x)
        if paddingBottom > 0 or paddingRight > 0: x = x[:, :, :H, :W]
        x = x + inp
        if return_features:
            return x, memory_list, tuple(feature_list)
        return x, memory_list

# ====================================================================================
# SECTION 3: 感知算子 & NAR_Net 主体
# ====================================================================================

class BaseKroneckerSensing(nn.Module):
    def __init__(self, H, W, max_cr, binary=True, normalize=True, binarization_mode='ste', initial_annealing_temp=1.0):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.max_cr = float(max_cr)
        self.binary = bool(binary)
        self.normalize = bool(normalize)
        self.binarization_mode = str(binarization_mode).lower()
        self.register_buffer('annealing_temp', torch.tensor(float(initial_annealing_temp), dtype=torch.float32))
        
        self.m_max = max(1, int(math.ceil(self.H * math.sqrt(self.max_cr))))
        self.n_max = max(1, int(math.ceil(self.W * math.sqrt(self.max_cr))))

    def _get_dimensions(self, cr):
        cr_val = self.max_cr if cr is None else float(cr)
        cr_val = float(max(1e-4, min(self.max_cr, cr_val)))
        m = max(1, int(math.ceil(self.H * math.sqrt(cr_val))))
        n = max(1, int(math.ceil(self.W * math.sqrt(cr_val))))
        return min(m, self.m_max), min(n, self.n_max)

    def _apply_binary_normalize(self, mat):
        eff = mat
        if self.binary:
            if self.binarization_mode == 'ste':
                mat_bin = torch.sign(mat)
                eff = mat + (mat_bin - mat).detach()
            else:
                k = float(cast(torch.Tensor, self.annealing_temp).item())
                eff = torch.tanh(k * mat)
                if not self.training: eff = torch.sign(eff)
        if self.normalize:
            eff = eff / eff.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        return eff

    def effective_params(self, cr=None): raise NotImplementedError
    def forward(self, X, cr=None, effective_params=None): raise NotImplementedError
    def adjoint(self, Y, cr=None, effective_params=None): raise NotImplementedError

    def prox_f(self, X, Y, rho, cr=None):
        eff_params = self.effective_params(cr)
        Y_hat = self.forward(X, cr, effective_params=eff_params)
        residual = Y - Y_hat
        grad = self.adjoint(residual, cr, effective_params=eff_params)
        if rho.dim() == 0: rho = rho.view(1,1,1,1)
        elif rho.dim() == 1: rho = rho.view(-1,1,1,1)
        return X + rho * grad

class KroneckerSensing(BaseKroneckerSensing):
    def __init__(self, H, W, max_cr, **kwargs):
        super().__init__(H, W, max_cr, **kwargs)
        self.Phi_stack = nn.Parameter(torch.randn(self.m_max, 1, self.H) / math.sqrt(max(1, self.H)))
        self.Psi_stack = nn.Parameter(torch.randn(self.n_max, 1, self.W) / math.sqrt(max(1, self.W)))

    def effective_params(self, cr=None):
        m, n = self._get_dimensions(cr)
        phi = self._apply_binary_normalize(self.Phi_stack[:m, :, :])
        psi = self._apply_binary_normalize(self.Psi_stack[:n, :, :])
        return phi, psi

    def forward(self, X, cr=None, effective_params=None):
        B, C, H, W = X.shape
        # 使用 reshape 代替 view，防止非连续 Tensor 报错
        Xb = X.reshape(B * C, H, W)
        if effective_params is None: phi, psi = self.effective_params(cr)
        else: phi, psi = effective_params
        
        tmp = torch.einsum('mh,bhw->bmw', phi.squeeze(1), Xb)
        Y = torch.einsum('bmw,nw->bmn', tmp, psi.squeeze(1))
        return Y.view(B, C, Y.shape[1], Y.shape[2])

    def adjoint(self, Y, cr=None, effective_params=None):
        B, C, m, n = Y.shape
        Yb = Y.reshape(B * C, m, n)
        if effective_params is None: phi, psi = self.effective_params(cr)
        else: phi, psi = effective_params
        
        tmp = torch.einsum('bmn,nw->bmw', Yb, psi.squeeze(1))
        Xb = torch.einsum('mh,bmw->bhw', phi.squeeze(1), tmp)
        return Xb.view(B, C, self.H, self.W)

class AsymmetricKroneckerSensing(BaseKroneckerSensing):
    def __init__(self, H, W, max_cr, **kwargs):
        super().__init__(H, W, max_cr, **kwargs)
        
        self.Phi_stack = nn.Parameter(torch.empty(self.m_max, 1, self.H))
        self.Psi_stack = nn.Parameter(torch.empty(self.m_max, self.n_max, self.W))

        with torch.no_grad():
            nn.init.normal_(self.Phi_stack, mean=0.0, std=1.0)
            self.Phi_stack.data /= math.sqrt(max(1, self.H))
            nn.init.normal_(self.Psi_stack, mean=0.0, std=1.0)
            self.Psi_stack.data /= math.sqrt(max(1, self.W))

    def effective_params(self, cr=None):
        m, n = self._get_dimensions(cr)
        phi = self._apply_binary_normalize(self.Phi_stack[:m, :, :])
        psi = self._apply_binary_normalize(self.Psi_stack[:m, :n, :])
        return phi, psi

    def forward(self, X, cr=None, effective_params=None):
        B, C, H, W = X.shape
        Xb = X.reshape(B * C, H, W)
        if effective_params is None: phi, psi = self.effective_params(cr)
        else: phi, psi = effective_params
        
        tmp = torch.einsum('mh,bhw->bmw', phi.squeeze(1), Xb)
        Y = torch.einsum('bmw,mnw->bmn', tmp, psi)
        return Y.view(B, C, Y.shape[1], Y.shape[2])

    def adjoint(self, Y, cr=None, effective_params=None):
        B, C, m, n = Y.shape
        Yb = Y.reshape(B * C, m, n)
        if effective_params is None: phi, psi = self.effective_params(cr)
        else: phi, psi = effective_params
        
        tmp = torch.einsum('bmn,mnw->bmw', Yb, psi)
        Xb = torch.einsum('mh,bmw->bhw', phi.squeeze(1), tmp)
        return Xb.view(B, C, self.H, self.W)

class NKCSSensing(BaseKroneckerSensing):
    """
    Nested Kronecker Compressive Sensing (NKCS) 感知算子
    核心机制:
    1. Master Sequence: 能量最高的 K_d (Head) 确定性排序，剩余 (Tail) 随机置换。
    2. Random Modulation: 仅对随机尾部 (i > K_d) 的列算子施加 Rademacher 符号调制。
    3. 行/列算子默认使用 DCT 正交基底，按能量贡献率排序。
    """
    @staticmethod
    def _dct_basis(N):
        """Generate NxN DCT-II orthonormal basis matrix, rows ordered by frequency (energy)."""
        n = torch.arange(N, dtype=torch.float64)
        k = torch.arange(N, dtype=torch.float64)
        # (N, N) matrix where [k, n] = cos(pi * k * (2n+1) / (2N))
        basis = torch.cos(math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
        basis[0, :] *= 1.0 / math.sqrt(N)
        basis[1:, :] *= math.sqrt(2.0 / N)
        return basis.float()

    def __init__(self, H, W, max_cr, head_ratio=0.05, **kwargs):
        super().__init__(H, W, max_cr, **kwargs)
        
        # 定义参数容器：Psi 维度为 (m, n, W)，支持每行独立的列调制
        self.Phi_stack = nn.Parameter(torch.empty(self.m_max, 1, self.H))
        self.Psi_stack = nn.Parameter(torch.empty(self.m_max, self.n_max, self.W))
        
        # --- 1. 使用 DCT 正交基底 (按能量/频率升序排列) ---
        basis_phi = self._dct_basis(self.H)  # (H, H)
        basis_psi = self._dct_basis(self.W)  # (W, W)

        # --- 2. 构建 NKCS 主序列 (Deterministic Head + Random Tail) ---
        # 保证 m 较小时只采样 Head，m 变大时自动扩展到 Tail (嵌套性)
        K = int(self.H * head_ratio)
        indices = np.arange(self.H)
        
        head_idx = indices[:K]          # Head: 严格保留能量最高的顺序 (低频 DCT 基底)
        tail_idx = indices[K:]          # Tail: 高频部分随机打乱
        np.random.shuffle(tail_idx)     # [NKCS 核心: 引入随机性以降低相干性]
        
        master_seq = np.concatenate([head_idx, tail_idx])
        
        with torch.no_grad():
            # 取出前 n_max 个水平 DCT 基底备用
            psi_base = basis_psi[:self.n_max, :]  # (n, W)

            # --- 3. 填充堆栈 ---
            for i in range(self.m_max):
                # 按序列取行索引 (处理 m_max > H 的循环情况)
                row_ptr = master_seq[i % self.H]
                
                # A. 填充行算子 Phi (按序列取值)
                self.Phi_stack.data[i, 0, :] = basis_phi[row_ptr, :]
                
                # B. 填充列算子 Psi
                # 论文公式 (5): 确定性头部使用纯基底向量，随机尾部施加 Rademacher 调制
                if i < K:
                    # 确定性头部 (i <= K_d): 不做 Rademacher 调制，保留低频能量
                    self.Psi_stack.data[i, :, :] = psi_base
                else:
                    # 随机尾部 (i > K_d): 行绑定 Rademacher 符号调制
                    # 使用确定性种子确保嵌套一致性: row_ptr 相同的行，调制必须相同
                    g = torch.Generator()
                    g.manual_seed(int(row_ptr) + 2024)
                    
                    # 生成随机符号 {-1, +1}
                    epsilon = torch.randint(0, 2, (1, self.W), generator=g).float() * 2 - 1
                    
                    # 广播调制: (n, W) * (1, W) -> (n, W)
                    self.Psi_stack.data[i, :, :] = psi_base * epsilon.to(psi_base.device)

            # 注入极微量噪声防止梯度死锁
            self.Phi_stack.data.add_(torch.randn_like(self.Phi_stack) * 0.001)

    # 复用 BaseKroneckerSensing 的逻辑 (需显式实现 forward/adjoint 因为父类只是接口)
    def effective_params(self, cr=None):
        m, n = self._get_dimensions(cr)
        # 这里的切片 [:m] 配合初始化时的 master_seq 排序，实现了 NKCS 的嵌套采样
        phi = self._apply_binary_normalize(self.Phi_stack[:m, :, :])
        psi = self._apply_binary_normalize(self.Psi_stack[:m, :n, :])
        return phi, psi

    def forward(self, X, cr=None, effective_params=None):
        B, C, H, W = X.shape
        Xb = X.reshape(B * C, H, W)
        phi, psi = effective_params if effective_params else self.effective_params(cr)
        
        # 1. Row Projection: (m, H) x (H, W) -> (m, W)
        tmp = torch.einsum('mh,bhw->bmw', phi.squeeze(1), Xb)
        # 2. Col Projection: (m, W) x (m, n, W)^T -> (m, n)
        # 注意：这里利用 'mnw' 实现了每一行使用不同的列基底 (NKCS Row-Dependent)
        Y = torch.einsum('bmw,mnw->bmn', tmp, psi)
        return Y.view(B, C, Y.shape[1], Y.shape[2])

    def adjoint(self, Y, cr=None, effective_params=None):
        B, C, m, n = Y.shape
        Yb = Y.reshape(B * C, m, n)
        phi, psi = effective_params if effective_params else self.effective_params(cr)
        
        tmp = torch.einsum('bmn,mnw->bmw', Yb, psi)
        Xb = torch.einsum('mh,bmw->bhw', phi.squeeze(1), tmp)
        return Xb.view(B, C, self.H, self.W)

class NAR_Net(nn.Module):
    def __init__(self, 
                 stages=6,
                 scale_factor=1,
                 max_cr=0.5,
                 color_channel=1,
                 sensing_mode='nkcs',
                 binary_sensing=True,
                 normalize_sensing=True,
                 binarization_mode='ste',
                 supported_lr_resolutions=None,
                 prior_config=None,
                 use_checkpoint=True):
        super(NAR_Net, self).__init__()
        
        self.K = stages
        if scale_factor != 1: print("## INFO: scale_factor forced to 1 for pure CS task. ##")
        self.scale_factor = 1
        self.max_cr = float(max_cr)
        self.color_channel = color_channel
        self.use_checkpoint = use_checkpoint
        self.sensing_mode = sensing_mode.lower()

        if prior_config is None:
            prior_config = {
                'color_channel': color_channel, 'dim': 48, 'head_dim': 16, 'window_size': 8,
                'enc_blocks': [2, 2, 2], 'mid_blocks': 2, 'dec_blocks': [2, 2, 2], 'hsb_sr': 2
            }
        
        self.prior_module = NAR_Prior(**prior_config)
        # 预计算 Memory 列表长度，用于 Checkpoint 输出重组
        self.num_mem = 1 + len(prior_config.get('dec_blocks', [2,2,2]))
        
        # 实例化 Embedding，维度需与 Prior 内部 dim 一致 (默认48或64)
        embed_dim = prior_config.get('dim', 48) # 需确认与 Prior配置一致
        self.cr_embedder = CREmbedding(dim=embed_dim)

        self.sensing_ops = nn.ModuleDict()
        resolutions_to_init = supported_lr_resolutions if supported_lr_resolutions else [(128, 128), (256, 256), (512, 512)]

        for res in resolutions_to_init:
            if isinstance(res, (list, tuple)) and len(res) == 2:
                H_lr, W_lr = int(res[0]), int(res[1])
                self._create_sensing_op(H_lr, W_lr, binary_sensing, normalize_sensing, binarization_mode)

        self.rho_cs_params = nn.Parameter(torch.full((self.K,), -2.25))
        self.gamma_params = nn.Parameter(torch.full((self.K,), -2.2))
        self.beta_params = nn.Parameter(torch.full((self.K,), -2.944))
        pt_mix = torch.ones(self.K); pt_mix[-1] = 0.0
        self.register_buffer('pt_mix_weights', pt_mix)

    def _create_sensing_op(self, H, W, binary, normalize, bin_mode, initial_temp=1.0):
        key = f"{H}x{W}"
        if key in self.sensing_ops: return
        
        # 感知模式判断
        if self.sensing_mode == 'nkcs':
            op = NKCSSensing(
                H, W, self.max_cr, 
                head_ratio=0.05,       # 设定 5% 为确定性低频头
                binary=binary, normalize=normalize, 
                binarization_mode=bin_mode, initial_annealing_temp=initial_temp
            )
        elif self.sensing_mode == 'standard':
            op = KroneckerSensing(H, W, self.max_cr, binary=binary, normalize=normalize, 
                                  binarization_mode=bin_mode, initial_annealing_temp=initial_temp)
        else:
            # AKCS (Asymmetric KCS)
            op = AsymmetricKroneckerSensing(
                H, W, self.max_cr,
                binary=binary, normalize=normalize, 
                binarization_mode=bin_mode, initial_annealing_temp=initial_temp
            )
        self.sensing_ops[key] = op

    def _ideal_prox_target(self, gt, prox_input, stage_idx):
        if gt is None: return prox_input
        mix = self.pt_mix_weights[stage_idx] 
        return (mix * prox_input + gt) / (mix + 1.0)

    def _run_prior_checkpoint(self, x, beta, return_features_flag, cr_emb, *mem_args):
        mem_in = list(mem_args) if mem_args else None
        return self.prior_module(x, mem_in, beta_k=beta, return_features=return_features_flag, cr_emb=cr_emb)

    def forward(self, GT_HR=None, Y=None, H_hr=None, W_hr=None, cr=None, return_features=False):
        current_cr = float(max(1e-4, min(self.max_cr, float(cr) if cr is not None else self.max_cr)))
        
        # 生成 Embedding 向量
        cr_emb = self.cr_embedder(current_cr) 

        if GT_HR is not None:
            if GT_HR.dim() == 3: GT_HR = GT_HR.unsqueeze(1)
            _, _, H_hr, W_hr = GT_HR.shape
            H_lr, W_lr = H_hr, W_hr
        elif Y is not None and H_hr is not None and W_hr is not None:
            H_lr, W_lr = H_hr, W_hr
        else:
            raise ValueError("GT_HR or Y info required.")

        key = f"{H_lr}x{W_lr}"
        if key not in self.sensing_ops:
            ref_op = next(iter(self.sensing_ops.values()))
            binary = getattr(ref_op, 'binary', True)
            normalize = getattr(ref_op, 'normalize', True)
            bin_mode = getattr(ref_op, 'binarization_mode', 'ste')
            temp = getattr(ref_op, 'annealing_temp', torch.tensor(1.0)).item()
            
            self._create_sensing_op(H_lr, W_lr, binary, normalize, bin_mode, temp)
            self.sensing_ops[key] = self.sensing_ops[key].to(GT_HR.device if GT_HR is not None else Y.device)

        sensing_op = self.sensing_ops[key]
        if Y is None: Y = sensing_op(GT_HR, cr=current_cr)
        X_k = sensing_op.adjoint(Y, cr=current_cr)
        M_k = [] # Init as empty list

        outputs_Xk, outputs_Zk, outputs_ProxIdeal = [], [], []
        outputs_Features = [] if return_features else None

        rho_cs = F.softplus(self.rho_cs_params)
        gamma_k_vals = torch.sigmoid(self.gamma_params)
        beta_k_vals = F.softplus(self.beta_params)

        for k in range(self.K):
            X_k_prev = X_k
            beta_k = beta_k_vals[k].view(1, 1, 1, 1)
            X_k_degraded = X_k_prev + torch.randn_like(X_k_prev) * beta_k

            if self.training and self.use_checkpoint:
                args = [X_k_degraded, beta_k_vals[k], return_features, cr_emb] + (M_k if M_k else [])
                try: flat_res = checkpoint(self._run_prior_checkpoint, *args, use_reentrant=False)
                except TypeError: flat_res = checkpoint(self._run_prior_checkpoint, *args)
            else:
                flat_res = self._run_prior_checkpoint(X_k_degraded, beta_k_vals[k], return_features, cr_emb, *M_k)

            if return_features:
                x_out, mem_out, feats_out = flat_res
                X_R_k = x_out
                M_k = list(mem_out)
                Features_k = feats_out
            else:
                x_out, mem_out = flat_res
                X_R_k = x_out
                M_k = list(mem_out)
                Features_k = None

            gamma_k = gamma_k_vals[k].view(1, 1, 1, 1)
            U_k = (1.0 - gamma_k) * X_k_prev + gamma_k * X_R_k
            outputs_Zk.append(U_k)

            X_k = sensing_op.prox_f(U_k, Y, rho_cs[k], cr=current_cr)
            outputs_Xk.append(X_k)
            outputs_ProxIdeal.append(self._ideal_prox_target(GT_HR, U_k, k) if GT_HR is not None else X_k)
            if outputs_Features is not None: outputs_Features.append(Features_k or tuple())

        X_final = X_k
        return X_final, (outputs_Xk, outputs_Zk, outputs_ProxIdeal, outputs_Features or [])