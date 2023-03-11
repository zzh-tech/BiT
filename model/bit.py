import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from einops import rearrange

try:
    import model.loss as loss
    from model.utils import pair, ckpt_convert, calc_model_params
    from model.arches import default_conv
except:
    import loss
    from utils import pair, ckpt_convert, calc_model_params
    from arches import default_conv


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.use_conv = use_conv
        if use_conv:
            self.conv_layer = nn.Sequential(
                default_conv(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                             groups=hidden_features),
                nn.GELU()
            )

    def forward(self, x, x_size):
        # x: B, H * W, C
        # x_size: H, W
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.use_conv:
            x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W).contiguous()
            x = self.conv_layer(x)
            x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_conv=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_conv

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if self.use_conv:
            self.qkv = default_conv(in_channels=dim, out_channels=dim * 3, kernel_size=3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_conv:
            self.proj = default_conv(in_channels=dim, out_channels=dim, kernel_size=3)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        H, W = x_size
        if self.use_conv:
            assert self.window_size[0] == self.window_size[1]
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.qkv(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            qkv = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.use_conv:
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.proj(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            x = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mlp=True, use_conv=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_mlp = use_mlp
        self.mlp_ratio = mlp_ratio
        self.use_conv = use_conv
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_conv=use_conv)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.use_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           use_conv=use_conv)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask, x_size=x_size)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device), x_size=x_size)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_mlp=True, use_conv=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_mlp=use_mlp,
                                 use_conv=use_conv)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB).
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_mlp=True, use_conv=False, ms=True):
        super(RSTB, self).__init__()
        self.window_size = window_size
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_mlp=use_mlp,
                                         use_conv=use_conv)
        self.ms = ms
        if ms:
            self.fuse_layer = default_conv(in_channels=3 * dim, out_channels=dim, kernel_size=1)
        self.last_layer = default_conv(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        if self.ms:
            H1, W1 = x_size
            # H2, W2 should be divisible by window_size
            H2, W2 = H1 // 2, W1 // 2
            H2, W2 = H2 - (H2 % self.window_size), W2 - (W2 % self.window_size)
            # H3, W3 should be divisible by window_size
            H3, W3 = H1 // 4, W1 // 4
            H3, W3 = H3 - (H3 % self.window_size), W3 - (W3 % self.window_size)

            x1 = rearrange(x, 'B (H W) C -> B C H W', H=H1, W=W1)
            x2 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)

            x1 = rearrange(x1, 'B C H W -> B (H W) C')
            res1 = self.residual_group(x1, x_size=(H1, W1))  # B, H1*W1, C
            res1 = rearrange(res1, 'B (H W) C -> B C H W', H=H1, W=W1)

            x2 = rearrange(x2, 'B C H W -> B (H W) C')
            res2 = self.residual_group(x2, x_size=(H2, W2))  # B, H2*W2, C
            res2 = rearrange(res2, 'B (H W) C -> B C H W', H=H2, W=W2)
            res2 = F.interpolate(res2, size=(H1, W1), mode='bilinear', align_corners=False)

            x3 = rearrange(x3, 'B C H W -> B (H W) C')
            res3 = self.residual_group(x3, x_size=(H3, W3))  # B, H3*W3, C
            res3 = rearrange(res3, 'B (H W) C -> B C H W', H=H3, W=W3)
            res3 = F.interpolate(res3, size=(H1, W1), mode='bilinear', align_corners=False)

            res = torch.cat([res1, res2, res3], dim=1)
            res = self.last_layer(self.fuse_layer(res))
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res
        else:
            H, W = x_size
            res = self.residual_group(x, x_size)  # B, H*W, C
            res = rearrange(res, 'B (H W) C -> B C H W', H=H, W=W)
            res = self.last_layer(res)
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res


class BiT(nn.Module):
    def __init__(self, img_size=256, in_chs=3, out_chs=3, embed_dim=64, depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4),
                 window_size=8, mlp_ratio=2, norm_layer=nn.LayerNorm, use_mlp=True, use_conv=False, num_mid=2,
                 in_num=2, skip=False, anchor_num_layers=3, ms=True):
        super(BiT, self).__init__()
        self.window_size = window_size
        self.num_mid = num_mid
        self.in_num = in_num
        self.skip = skip
        self.anchor_num_layers = anchor_num_layers

        #  shallow feature extraction
        self.conv_down = nn.Sequential(default_conv(in_chs, embed_dim // in_num, kernel_size=3, stride=2),
                                       nn.GELU(),
                                       default_conv(embed_dim // in_num, embed_dim // in_num, kernel_size=3, stride=2))
        self.linear_reduce = nn.Linear(embed_dim + 1, embed_dim)

        # deep feature extraction
        self.num_layers = len(depths)  # number of Swin basic layers
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.layers = nn.ModuleList()
        Ph = Pw = img_size // window_size
        input_resolution = (Ph, Pw)
        for i in range(self.num_layers):
            self.layers.append(
                RSTB(
                    dim=embed_dim, input_resolution=input_resolution, depth=depths[i], num_heads=num_heads[i],
                    window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm, use_checkpoint=False,
                    use_mlp=use_mlp, use_conv=use_conv, ms=ms
                )
            )
        self.norm = norm_layer(self.embed_dim)

        # reconstruction
        self.mid_shuffle_up = nn.Sequential(
            default_conv(embed_dim, num_mid * out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4))
        self.last_shuffle_up = nn.Sequential(default_conv(embed_dim, out_chs * (4 ** 2), kernel_size=3, stride=1),
                                             nn.PixelShuffle(4))

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def pad_image(self, x, pad_size):
        C, H, W = x.size()[-3:]
        pad_h = (pad_size - H % pad_size) % pad_size
        pad_w = (pad_size - W % pad_size) % pad_size
        if len(x.shape) == 5:
            N = x.shape[1]
            x = F.pad(x.reshape(-1, C, H, W), (0, pad_w, 0, pad_h), 'reflect')
            x = x.reshape(-1, N, *x.shape[-3:])
        elif len(x.shape) == 4:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        else:
            raise ValueError
        return x

    def forward(self, x, t):
        # x: B, 2, C', H', W'
        # t: 0~1
        Hi, Wi = x.shape[-2:]
        x = self.pad_image(x, 4 * self.window_size)
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        x = self.conv_down(x)
        x = rearrange(x, '(B N) C H W -> B (N C) H W', N=self.in_num)
        H, W = x.shape[-2:]
        x_size = (H, W)
        x = rearrange(x, 'B C H W -> B (H W) C')

        if self.anchor_num_layers > 0:
            res = x
            for i in range(self.anchor_num_layers):
                layer = self.layers[i]
                x = layer(x, x_size)
            if self.skip:
                x = res + x
        mid_x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        mid_x = self.mid_shuffle_up(mid_x)
        mid_x = rearrange(mid_x, 'B (N C) H W -> B N C H W', N=self.num_mid)
        mid_x = mid_x[:, :, :, :Hi, :Wi]

        t = torch.ones_like(x[:, :, :1]) * t.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = torch.cat([x, t], dim=-1)  # B, H*W, C+1
        x = self.linear_reduce(x)  # B, H*W, C
        res = x
        for i in range(self.anchor_num_layers, self.num_layers):
            layer = self.layers[i]
            x = layer(x, x_size)
        if self.skip:
            x = res + x
        x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        x = self.last_shuffle_up(x)

        last_x = x[:, :, :Hi, :Wi]
        return last_x, mid_x

    def multi_forward(self, x, ts):
        # x: B, 2, C', H', W'
        # ts: [0, ..., 1]
        Hi, Wi = x.shape[-2:]
        x = self.pad_image(x, 4 * self.window_size)
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        x = self.conv_down(x)
        x = rearrange(x, '(B N) C H W -> B (N C) H W', N=self.in_num)
        H, W = x.shape[-2:]
        x_size = (H, W)
        x = rearrange(x, 'B C H W -> B (H W) C')

        res = x
        for i in range(self.anchor_num_layers):
            layer = self.layers[i]
            x = layer(x, x_size)
        if self.skip:
            x = res + x
        mid_x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        mid_x = self.mid_shuffle_up(mid_x)
        mid_x = rearrange(mid_x, 'B (N C) H W -> B N C H W', N=self.num_mid)
        mid_x = mid_x[:, :, :, :Hi, :Wi]

        last_x = []
        for t in ts:
            t = torch.ones_like(x[:, :, :1]) * t
            out = torch.cat([x, t], dim=-1)  # B, H*W, C+1
            out = self.linear_reduce(out)  # B, H*W, C
            res = out
            for i in range(self.anchor_num_layers, self.num_layers):
                layer = self.layers[i]
                out = layer(out, x_size)
            if self.skip:
                out = res + out
            out = rearrange(out, 'B (H W) C -> B C H W', H=H, W=W)
            out = self.last_shuffle_up(out)
            last_x.append(out)
        last_x = torch.stack(last_x, dim=1)
        last_x = last_x[:, :, :, :Hi, :Wi]
        return last_x, mid_x

    def inference(self, x, ts):
        # x: B, 2, C', H', W'
        # ts: [0, ..., 1]
        Hi, Wi = x.shape[-2:]
        x = self.pad_image(x, 4 * self.window_size)
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        x = self.conv_down(x)
        x = rearrange(x, '(B N) C H W -> B (N C) H W', N=self.in_num)
        H, W = x.shape[-2:]
        x_size = (H, W)
        x = rearrange(x, 'B C H W -> B (H W) C')

        if self.anchor_num_layers > 0:
            res = x
            for i in range(self.anchor_num_layers):
                layer = self.layers[i]
                x = layer(x, x_size)
            if self.skip:
                x = res + x

        last_x = []
        for t in ts:
            t = torch.ones_like(x[:, :, :1]) * t
            out = torch.cat([x, t], dim=-1)  # B, H*W, C+1
            out = self.linear_reduce(out)  # B, H*W, C
            res = out
            for i in range(self.anchor_num_layers, self.num_layers):
                layer = self.layers[i]
                out = layer(out, x_size)
            if self.skip:
                out = res + out
            out = rearrange(out, 'B (H W) C -> B C H W', H=H, W=W)
            out = self.last_shuffle_up(out)
            last_x.append(out)
        last_x = torch.stack(last_x, dim=1)
        last_x = last_x[:, :, :, :Hi, :Wi]
        return last_x

    def inference_feat(self, x, ts):
        # x: B, 2, C', H', W'
        # ts: [0, ..., 1]
        Hi, Wi = x.shape[-2:]
        x = self.pad_image(x, 4 * self.window_size)
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        x = self.conv_down(x)
        x = rearrange(x, '(B N) C H W -> B (N C) H W', N=self.in_num)
        H, W = x.shape[-2:]
        x_size = (H, W)
        x = rearrange(x, 'B C H W -> B (H W) C')

        if self.anchor_num_layers > 0:
            res = x
            for i in range(self.anchor_num_layers):
                layer = self.layers[i]
                x = layer(x, x_size)
            if self.skip:
                x = res + x

        share_feat = x
        mid_feats = []
        pred_feats = []
        for t in ts:
            t = torch.ones_like(share_feat[:, :, :1]) * t
            out = torch.cat([share_feat, t], dim=-1)  # B, H*W, C+1
            out = self.linear_reduce(out)  # B, H*W, C
            mid_feats.append(rearrange(out, 'B (H W) C -> B C H W', H=H, W=W))

            res = out
            for i in range(self.anchor_num_layers, self.num_layers):
                layer = self.layers[i]
                out = layer(out, x_size)
            if self.skip:
                out = res + out
            out = rearrange(out, 'B (H W) C -> B C H W', H=H, W=W)
            pred_feats.append(out)

        mid_feats = torch.stack(mid_feats, dim=1)
        pred_feats = torch.stack(pred_feats, dim=1)
        mid_feats = mid_feats[:, :, :, :Hi, :Wi]
        pred_feats = pred_feats[:, :, :, :Hi, :Wi]
        share_feat = rearrange(share_feat, 'B (H W) C -> B C H W', H=H, W=W)
        return mid_feats, pred_feats, share_feat


class Model:
    def __init__(self, image_size, in_chs, out_chs, embed_dim, depths, num_heads,
                 window_size, mlp_ratio, num_mid, optimizer_args, scheduler_args, loss_args,
                 num_gts=3, use_mlp=True, use_conv=False, norm_layer=nn.LayerNorm, ms=True,
                 in_num=2, skip=False, anchor_num_layers=3, mid_ratio=1.,
                 local_rank=0, load_from=None, resume_from=None, ddp=True):
        super(Model, self).__init__()
        self.num_gts = num_gts
        self.num_mid = num_mid
        self.mid_ratio = mid_ratio
        self.model = BiT(img_size=image_size, in_chs=in_chs, out_chs=out_chs, embed_dim=embed_dim, depths=depths,
                         num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer, use_mlp=use_mlp, use_conv=use_conv, num_mid=num_mid,
                         in_num=in_num, skip=skip, anchor_num_layers=anchor_num_layers, ms=ms)
        if ddp:
            # replace BN as SyncBN
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # load checkpoint
        if (local_rank == 0) and (load_from is not None):
            self.load_model(load_from)
        # resume checkpoint
        if (local_rank == 0) and (resume_from is not None):
            self.load_model(resume_from)
        self.device = torch.device("cuda", local_rank)
        # move model to GPU
        self.model = self.model.to(self.device)
        # nulti-gpus wrapper
        if ddp:
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True,
                             broadcast_buffers=False)
        else:
            self.model = DP(self.model)

        # Initialize optimizer, learning rate scheduler, and loss function
        self.optimizer = getattr(optim, optimizer_args['name'])(self.model.parameters(), **optimizer_args['args'])
        self.scheduler = getattr(optim.lr_scheduler, scheduler_args['name'])(self.optimizer, **scheduler_args['args'])
        if (local_rank == 0) and (resume_from is not None):
            self.load_others(resume_from)
        try:
            self.loss = getattr(nn, loss_args['name'])(**loss_args['args'])
        except:
            self.loss = getattr(loss, loss_args['name'])(**loss_args['args'])

    def load_model(self, load_from):
        model_dict = torch.load(load_from)
        self.model.load_state_dict(ckpt_convert(model_dict['model']))

    def load_others(self, load_from):
        model_dict = torch.load(load_from)
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.scheduler.load_state_dict(model_dict['scheduler'])

    def save_model(self, save_to):
        model_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(model_dict, save_to)

    def scheduler_step(self):
        self.scheduler.step()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_num_params(self):
        return calc_model_params(self.model)

    def update(self, inputs, training=True):
        results = inputs
        lq_imgs, gt_imgs = results['lq_imgs'], results['gt_imgs']  # B, N', C, H, W
        lq_img = lq_imgs[:, 1]
        mid_ts = torch.linspace(start=0, end=self.num_gts - 1, steps=self.num_mid, device=self.device).long()
        mid_gt_imgs = gt_imgs[:, mid_ts]  # B, num_mid, C, H, W
        gt_img = gt_imgs[:, self.num_gts // 2]

        if training:
            self.model.train()
            t = inputs['t'].to(self.device)
            pred_img, mid_pred_imgs = self.model(lq_imgs, t)  # B, C, H, W; B, num_mid, C, H, W
            mid_pred_imgs = rearrange(mid_pred_imgs, 'B N C H W -> (B N) C H W')
            mid_gt_imgs = rearrange(mid_gt_imgs, 'B N C H W -> (B N) C H W')
            loss_val = self.mid_ratio * (self.loss(mid_pred_imgs, mid_gt_imgs) / self.num_mid)
            loss_val = loss_val + self.loss(pred_img, gt_img)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor([0.5]).to(self.device)
                pred_img, mid_pred_imgs = self.model(lq_imgs, t=t)
                mid_pred_imgs = rearrange(mid_pred_imgs, 'B N C H W -> (B N) C H W')
                mid_gt_imgs = rearrange(mid_gt_imgs, 'B N C H W -> (B N) C H W')
                loss_val = self.mid_ratio * (self.loss(mid_pred_imgs, mid_gt_imgs) / self.num_mid)
                loss_val = loss_val + self.loss(pred_img, gt_img)
        results['pred_img'] = pred_img
        results['gt_img'] = gt_img
        results['lq_img'] = lq_img
        results['loss'] = loss_val
        return results

    @torch.no_grad()
    def inference(self, lq_imgs, ts):
        if isinstance(ts, list):
            ts = torch.tensor(ts)
        ts = ts.to(self.device)
        self.model.eval()
        pred_img = self.model.module.inference(lq_imgs, ts)
        return pred_img

    @torch.no_grad()
    def inference_feat(self, lq_imgs, ts):
        if isinstance(ts, list):
            ts = torch.tensor(ts)
        ts = ts.to(self.device)
        self.model.eval()
        mid_feats, pred_feats, share_feat = self.model.module.inference_feat(lq_imgs, ts)
        return mid_feats, pred_feats, share_feat
