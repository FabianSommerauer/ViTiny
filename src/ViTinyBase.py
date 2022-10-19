import torch
from torch import nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# Vision Transformer Implementation based on https://github.com/lucidrains/vit-pytorch

def position_encoding_sincos_2d(patches, temperature=10000):
    _, h, w, dim = patches.shape
    device = patches.device

    # generate 2d matrix of y & x patch positions
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    # calculate angular velocity for sines & cosines of embedding
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # multiply position coordinate of patch with each omega
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # positional embedding is given by combination of sin & cos of both x & y
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(patches.dtype)


# 2 layer perceptron dim -> hidden_dim -> dim
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # todo options for other
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.out(x)


# multiheaded self-attention
class MSA(nn.Module):
    def __init__(self, dim, heads=4, head_dim=16):
        super().__init__()
        total_head_dim = head_dim * heads
        self.head_count = heads
        self.head_dim = head_dim

        self.layer_norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, 3 * total_head_dim, bias=False)
        self.msa_out = nn.Linear(total_head_dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_scale_correction = self.head_dim ** -0.5

    def forward(self, x):
        x = self.layer_norm(x)

        qkv = self.qkv(x)

        # this is combines chunk(3, dim=-1) + map with rearrange
        # separate q k v and put split results for each head by a separate dimension
        # also make sure last dimensions are n d (since we want to compute attention between the n input vectors)
        q, k, v = rearrange(qkv, "b n (qkv_dim h d) -> qkv_dim b h n d", qkv_dim=3, h=self.head_dim)

        # swap dims n & d of k -> we get a  nxd * dxn = nxn matrix  (for each batch and head)
        att_dot = torch.matmul(q, k.transpose(-1, -2)) * self.attention_scale_correction

        attn = self.softmax(att_dot)

        # calculates nxn * nxd -> nxd matrix  (for each batch and head)
        weighted_values = torch.matmul(attn, v)

        # recombine the results of each head
        weighted_values = rearrange(weighted_values, "b h n d -> b n (h d)")

        # apply linear layer to results of each head
        return self.msa_out(weighted_values)


class Transformer(nn.Module):
    def __init__(self, dim, depth, head_count, head_dim, mlp_dim):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MSA(dim, head_count, head_dim),
                MLP(dim, mlp_dim)
            ]))

    def forward(self, x):
        for msa, mlp in self.layers:
            # apply msa & mlp with residual connection (layer norm will fix variance/bias issues)
            x = msa(x) + x
            x = mlp(x) + x

        return x


class ViTinyBase(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, depth, dim, mlp_dim, head_count=4, head_dim=16, channels=3):
        super().__init__()

        height, width = image_size
        patch_height, patch_width = patch_size

        assert height % patch_height == 0 and width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        patch_count = (width // patch_width) * (height // patch_height)
        patch_size = channels * patch_width * patch_height

        self.embed_image = nn.Sequential(
            # this splits the image into patches
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_size, dim)
        )

        self.transformer = Transformer(dim, depth, head_count, head_dim, mlp_dim)

        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            # todo: remove softmax if we need output for something else than classification (e.g. for Masked pre-training)
            nn.Softmax(dim=-1)
        )

    def forward(self, img):
        *_, h, w = img.shape

        x = self.embed_image(img)

        pos_encode = position_encoding_sincos_2d(x)  # todo: maybe test different temperatures?

        # x is now in format b h w dim  (h & w = locations of patches)
        # combine h & w into single dim
        x = rearrange(x, "b ... dim -> b (...) dim")

        x = x + pos_encode

        x = self.transformer(x)

        # compute mean result over all patches
        x = reduce(x, "b n dim -> b dim", 'mean')

        return self.out(x)


