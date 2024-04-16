import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from DiffIR.archs import basicblock as B
import functools
import numpy as np
from einops import rearrange, repeat
from basicsr.utils.registry import ARCH_REGISTRY

#   UNetDiscriminatorSN Discriminator_PatchGAN  Discriminator_VGG_96    Discriminator_VGG_128   Discriminator_VGG_192   Discriminator_VGG_128_SN    Discriminator_ViT

@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


# --------------------------------------------
# PatchGAN discriminator
# If n_layers = 3, then the receptive field is 70x70
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_PatchGAN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, n_layers=3, norm_type='spectral'):
        '''PatchGAN discriminator, receptive field = 70x70 if n_layers = 3
        Args:
            input_nc: number of input channels
            ndf: base channel number
            n_layers: number of conv layer with stride 2
            norm_type:  'batch', 'instance', 'spectral', 'batchspectral', instancespectral'
        Returns:
            tensor: score
        '''
        super(Discriminator_PatchGAN, self).__init__()
        self.n_layers = n_layers
        norm_layer = self.get_norm_layer(norm_type=norm_type)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[self.use_spectral_norm(nn.Conv2d(num_in_ch, num_feat, kernel_size=kw, stride=2, padding=padw), norm_type), nn.LeakyReLU(0.2, True)]]

        nf = num_feat
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_type),
                          norm_layer(nf),
                          nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_type),
                      norm_layer(nf),
                      nn.LeakyReLU(0.2, True)]]

        sequence += [[self.use_spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), norm_type)]]

        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.add_module('child' + str(n), nn.Sequential(*sequence[n]))

        self.model.apply(self.weights_init)

    def use_spectral_norm(self, module, norm_type='spectral'):
        if 'spectral' in norm_type:
            return spectral_norm(module)
        return module

    def get_norm_layer(self, norm_type='instance'):
        if 'batch' in norm_type:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif 'instance' in norm_type:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.Identity)
        return norm_layer

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------
# VGG style Discriminator with 96x96 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_96(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, ac_type='BL'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv(num_in_ch, num_feat, kernel_size=3, mode='C')
        conv1 = B.conv(num_feat, num_feat, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 64
        conv2 = B.conv(num_feat, num_feat*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(num_feat*2, num_feat*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 128
        conv4 = B.conv(num_feat*2, num_feat*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(num_feat*4, num_feat*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 256
        conv6 = B.conv(num_feat*4, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv8 = B.conv(num_feat*8, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 128x128 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_128(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, ac_type='BL'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv(num_in_ch, num_feat, kernel_size=3, mode='C')
        conv1 = B.conv(num_feat, num_feat, kernel_size=4, stride=2, mode='C'+ac_type)
        # 64, 64
        conv2 = B.conv(num_feat, num_feat*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(num_feat*2, num_feat*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 32, 128
        conv4 = B.conv(num_feat*2, num_feat*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(num_feat*4, num_feat*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 16, 256
        conv6 = B.conv(num_feat*4, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 8, 512
        conv8 = B.conv(num_feat*8, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 192x192 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_192(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, ac_type='BL'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv(num_in_ch, num_feat, kernel_size=3, mode='C')
        conv1 = B.conv(num_feat, num_feat, kernel_size=4, stride=2, mode='C'+ac_type)
        # 96, 64
        conv2 = B.conv(num_feat, num_feat*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(num_feat*2, num_feat*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 128
        conv4 = B.conv(num_feat*2, num_feat*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(num_feat*4, num_feat*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 256
        conv6 = B.conv(num_feat*4, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 512
        conv8 = B.conv(num_feat*8, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv10 = B.conv(num_feat*8, num_feat*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv11 = B.conv(num_feat*8, num_feat*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# SN-VGG style Discriminator with 128x128 input
# --------------------------------------------
@ARCH_REGISTRY.register()
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

@ARCH_REGISTRY.register()
class Discriminator_ViT(nn.Module):
    def __init__(self,
        num_in_ch = 3,
        num_feat=64,
        patch_size = 8,
        extend_size = 2,
        dim = 384,
        blocks = 6,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Discriminator_ViT, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        self.token_dim = num_in_ch * (self.patch_size ** 2)
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)


    def forward(self, img):
        # Generate overlappimg image patches
        stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        img_patches = img_patches.contiguous().view(
            img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        )
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim = 1)

        # Plus the positional embedding
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)

        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits