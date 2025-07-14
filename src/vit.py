import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from linformer_pytorch import LinearAttentionHead
# from . import register_cls_models


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.head = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.head
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k ,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qk)
        v = self.to_v(x).unsqueeze(1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)
        return out

# depth encdoer层数 ,heads 头个数 ,dim_head即dk
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    # Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    PerNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            # x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        feature_height, feature_width = pair(config['feature_size'])
        patch_height, patch_width = pair(config['patch_size'])
        assert feature_height % patch_height == 0 and feature_width % patch_width == 0
        num_patches = (feature_height // patch_height) * (feature_width % patch_width)
        patch_dim = config['channels'] * patch_height * patch_width

        assert config['pool'] in {'cls', 'mean'}
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, config['dim'])
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, config['dim']), requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['dim']), requires_grad=True)
        self.dropout = nn.Dropout(config['emb_dropout'])
        self.transformer = Transformer(config['dim'], config['depth'], config['heads'], config['dim_head'],
                                       config['mlp_dim'], config['dropout'])
        self.pool = config['pool']
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config['dim']),
            nn.Linear(config['dim'], config['num_classes']),
            nn.Sigmoid()
        )

    def forward(self, feature):
        x = self.to_patch_embedding(feature)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)