import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia
from einops import rearrange
import argparse
import math
import json
import numpy as np

from ldm.modules.id_embedding.iresnet import iresnet100
from ldm.modules.diffusionmodules.model import Normalize
from ldm.modules.e4e.encoders.psp_encoders import Encoder4Editing
from ldm.modules.e4e.psp import pSp
# from ldm.modules.e4e_human.psp import pSp as pSp_human
from ldm.modules.e4e.stylegan2.model import ModulatedConv2d, ScaledLeakyReLU
from  ldm.modules.e4e.stylegan2.op import FusedLeakyReLU
from ldm.modules.diffusionmodules.openaimodel import ResNet_timestep_embedder
from ldm.modules.diffusionmodules.util import linear, timestep_embedding
from ldm.modules.attention import BasicTransformerBlock


# ############### Copied from StyleGAN ################## #
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True, pre_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        self.non_linear = leaky_relu()

        self.lr_mul = lr_mul

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.LayerNorm(in_dim, eps=1e-5)

    def forward(self, input):
        if hasattr(self, 'pre_norm') and self.pre_norm:
            out = self.norm(input)
            out = F.linear(out, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        out = self.non_linear(out)
        return out


class Residual(nn.Module):
    def __init__(self,
                 fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class StyleVectorizer(nn.Module):
    def __init__(self, dim_in, dim_out, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            # if i == 0:
            #     layers.extend([EqualLinear(dim_in, dim_in // 2, lr_mul,)])
            # elif i == 1:
            #     layers.extend([EqualLinear(dim_in // 2, dim_out, lr_mul, pre_norm=True)])
            if i == 0:
                layers.extend([EqualLinear(dim_in, dim_out, lr_mul,)])
            else:
                layers.extend([Residual(EqualLinear(dim_out, dim_out, lr_mul, pre_norm=True))])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = F.normalize(x, dim=1)  # already normed at FR net
        return self.net(x)
    
class HspaceCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, int((3/4)*in_dim), 3, 1)
        self.conv2 = nn.Conv2d(int((3/4)*in_dim), out_dim, 3, 1)

    def forward(self, x):
        feat1 = nn.functional.leaky_relu(self.conv1(x))
        feat2 = nn.functional.leaky_relu(self.conv2(feat1))
        final_feat = nn.functional.adaptive_avg_pool2d(feat2, (1, 1))
        return final_feat
    
# NOTE: using different activation than the one in stylegan
class stylegan_based_mapper(nn.Module):
    def __init__(self, in_dim, out_dim, w_dim= 512, use_timestep_embedder=True):
        super().__init__()
        self.use_timesteps = use_timestep_embedder
        self.conv0 = nn.Conv2d(1280, in_dim, 3, 1, 1)
        self.modulated_conv1 = ModulatedConv2d(in_dim, in_dim, 3, 512)
        self.modulated_conv2 = ModulatedConv2d(in_dim, in_dim, 3, 512)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 3, 1)
        self.conv4 = nn.Conv2d(out_dim, out_dim, 3, 1)
        self.conv5 = nn.Conv2d(out_dim, out_dim, 3, 1)
        self.scaled_leaky_relu = ScaledLeakyReLU(0.2)
        self.self_attn = BasicTransformerBlock(512, 8, 64, 0.0)
        if(use_timestep_embedder):
            self.embed_time_w = ResNet_timestep_embedder(w_dim, 512, 0.1, 512)
            self.time_embed = nn.Sequential(
                linear(512, 512),
                nn.SiLU(),
                linear(512, 512),
            )
        
    def forward(self, h_space_feat, w_latents, ts=None):
        w_latent = self.self_attn(w_latents.view(w_latents.shape[0], 18, 512)).mean(dim=1)
        h_space_feat = nn.functional.leaky_relu(self.conv0(h_space_feat))
        if(self.use_timesteps):
            t_emb = timestep_embedding(ts, 512)
            t_emb = self.time_embed(t_emb)
            w_latent = self.embed_time_w(w_latent, t_emb)

        x = self.scaled_leaky_relu(self.modulated_conv1(h_space_feat, w_latent))
        x = self.scaled_leaky_relu(self.modulated_conv2(x, w_latent))

        x = nn.functional.leaky_relu(self.conv3(x))
        x = nn.functional.leaky_relu(self.conv4(x))
        x = nn.functional.leaky_relu(self.conv5(x))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x

class mapper_w_temb(nn.Module):
    def __init__(self, in_dim, out_dim, w_dim= 512, use_timestep_embedder=True, depth=4):
        super().__init__()
        self.use_timesteps = use_timestep_embedder
        self.final_mlp = StyleVectorizer(18*512 + 512, out_dim, depth, lr_mul=1.0) if use_timestep_embedder else StyleVectorizer(18*512, out_dim, depth, lr_mul=1.0)
        self.self_attn = BasicTransformerBlock(512, 8, 64, 0.0)
        if(use_timestep_embedder):
            # self.embed_time_w = ResNet_timestep_embedder(w_dim, 512, 0, 512)
            self.time_embed = nn.Sequential(
                linear(512, 512),
                nn.SiLU(),
                linear(512, 512),
            )
        
    def forward(self, h_space,  w_latents, ts=None):
        w_latent = self.self_attn(w_latents.view(w_latents.shape[0], 18, 512)).view(w_latents.shape[0], -1)

        if(self.use_timesteps):
            t_emb = timestep_embedding(ts, 512)
            t_emb = self.time_embed(t_emb)
            w_latent = torch.cat((w_latent, t_emb), dim=1)

        x = self.final_mlp(w_latent)
        return x


# ############### Copied from StyleGAN (END) ################## #

def normalize_basis_components(basis):
    basis = basis / torch.norm(basis, dim=2, keepdim=True)
    return basis

class VectorNorm(nn.Module):
    def __init__(self, dim=1, p=2):
        super(VectorNorm, self).__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)


class VectorSumAs(nn.Module):
    def __init__(self, norm_shape, dim=1, s=1.):
        super(VectorSumAs, self).__init__()
        self.norm_func = nn.BatchNorm1d(norm_shape, eps=1e-05)
        self.dim = dim
        self.s = s

    def forward(self, x):
        x = self.norm_func(x)
        # x = x - x.mean(dim=self.dim)
        return self.s * (x / x.sum(dim=self.dim, keepdims=True))


class MetaIdNet(nn.Module):
    def __init__(self,
                 fr_dim: int = 512,
                 meta_dim: int = 768,
                 inner_dim: int = 512,
                 context_dim: int = 768,
                 mlp_depth: int = 4,
                 use_expert: bool = False,
                 num_ids: int = 10,
                 expert_dim: int = 128,
                 use_header: bool = False,
                 use_celebs: bool = False,
                 num_embeds_per_token: int = 2,
                 heads: int = 1,
                 use_rm_mlp: bool = False,
                 vis_mean: bool = False,
                 vis_mean_params: tuple = None,
                 domain_clip_embed: torch.Tensor = None,
                 norm_reg_embedding: torch.Tensor = None,
                 use_stylegan_based_mapper: bool = True,
                 use_hspace: bool = False,
                 use_timestep_embedder: bool = True,
                 use_celeb_basis: bool = False,
                 use_basis_offset: bool = False,
                 celeb_basis: torch.Tensor = None,
                 shift_basis: bool = False,
                 use_regularization: bool = False,
                 domain_name="face",
                 ):
        super(MetaIdNet, self).__init__()
        self.num_ids = num_ids
        self.fr_dim = fr_dim
        self.meta_dim = meta_dim
        self.context_dim = context_dim
        self.num_es = num_embeds_per_token
        self.heads = heads
        self.vis_mean = vis_mean
        self.vis_mean_params = vis_mean_params
        self.domain_clip_embed = torch.cat([domain_clip_embed]*num_embeds_per_token, dim=0) if domain_clip_embed is not None else None
        self.norm_reg_embedding = norm_reg_embedding
        self.use_basis_offset = use_basis_offset
        self.use_regularization = use_regularization
        self.domain_name = domain_name

        if(shift_basis):   
            print("shifting mapper output accordin to basis")
            self.shift_basis = json.load(open('./weights/gaussian_approx_of_reconstructed_pca_weights.json', 'r'))
            basis_mean_token1 = self.shift_basis['token1']['mean_list']
            basis_mean_token2 = self.shift_basis['token2']['mean_list']
            basis_var_token1 = self.shift_basis['token1']['var_list']
            basis_var_token2 = self.shift_basis['token2']['var_list']
            scale_factor = np.array([basis_var_token1, basis_var_token2], dtype=np.float32)
            shift_factor = np.array([basis_mean_token1, basis_mean_token2], dtype=np.float32)
            self.scale_factor = torch.tensor(scale_factor, dtype=torch.float32).unsqueeze(1)
            self.shift_factor = torch.tensor(shift_factor, dtype=torch.float32).unsqueeze(1)
        else:
            self.shift_basis = None

        if(use_basis_offset):
            print("celeb basis shape", celeb_basis.shape)
            # self.basis_offset = nn.Parameter(celeb_basis[:, 1:], requires_grad=True)
            self.basis_offset = nn.Parameter(torch.randn(celeb_basis[:, 1:].shape), requires_grad=True)
            self.scale_factor = nn.Parameter(torch.zeros(512), requires_grad=True)
            self.shift_factor = nn.Parameter(torch.zeros(512), requires_grad=True)

        self.id_model = None
        # self.load_fr_net()
        # switched to dataloader
        if(self.domain_name=="face"):
            self.load_e4e_with_decoder('./weights/encoder/e4e_ffhq_encode.pt')
        elif(self.domain_name in ["human", 'dress']):
            # self.initilize_human_e4e("./weights/e4e_human.pt")
            pass

        self.register_buffer(
            name="trans_matrix",
            tensor=torch.tensor(
                [
                    [
                        [1.07695457, -0.03625215, -1.56352194 / 512],
                        [0.03625215, 1.07695457, -5.32134629 / 512],
                    ]
                ],
                requires_grad=False,
            ).float(),
        )  # (1,2,3) # a horrible bug if not '/512', difference between Pytorch grid_sample and Kornia warp_affine

        # ''' stylegan_mlp is the best '''
        # self.stylegan_mlp = StyleVectorizer(self.fr_dim, inner_dim * self.num_es * self.heads,
        #                                     depth=mlp_depth, lr_mul=1.0)
        self.use_stylegan_based_mapper = use_stylegan_based_mapper
        self.use_timestep_embedder = use_timestep_embedder
        if(self.use_stylegan_based_mapper):
            print("using wt based mapper")
            # self.stylegan_mlp = stylegan_based_mapper(512, 768 * self.num_es * self.heads, use_timestep_embedder=self.use_timestep_embedder)
            if(use_celeb_basis):
                self.stylegan_mlp = mapper_w_temb(18*512, 512 * self.num_es * self.heads, use_timestep_embedder=self.use_timestep_embedder)
            else:
                if(self.norm_reg_embedding is not None):
                    self.stylegan_mlp = mapper_w_temb(18*512, self.context_dim * self.num_es * self.heads, use_timestep_embedder=self.use_timestep_embedder,
                                                      depth=mlp_depth)
                else:
                    self.stylegan_mlp = mapper_w_temb(18*512, self.context_dim * self.num_es * self.heads, use_timestep_embedder=self.use_timestep_embedder
                                                      , depth=mlp_depth)

        else:
            if(use_hspace):
                print("using mapper with hspace")
                self.stylegan_mlp = StyleVectorizer(18 * 512, self.num_es * 512,
                                                    depth=mlp_depth, lr_mul=1.0)
                
                self.final_mlp = StyleVectorizer(self.num_es * 512 + 512, self.num_es * self.context_dim,
                                                    depth=mlp_depth, lr_mul=1.0)
                
                self.hspace_cnn = HspaceCNN(1280, 512)
            else:
                print("using mapper without hspace and timestep embedder")
                if(use_celeb_basis):
                    self.stylegan_mlp = StyleVectorizer(18 * 512, self.num_es * 512,
                                                    depth=mlp_depth, lr_mul=1.0)
                else:
                    self.stylegan_mlp = StyleVectorizer(18 * 512, self.num_es * 768,
                                                    depth=mlp_depth, lr_mul=1.0)

        print("[Number of layer in stylegan_mlp] ", mlp_depth)

        ''' expert mlp is not used '''
        self.use_expert = use_expert
        self.experts = []
        if use_expert:
            for idx in range(self.num_ids):
                expert = MlpBlock(self.meta_dim, expert_dim, self.meta_dim)
                self.experts.append(expert)

        ''' classification header is used for loss calculation, not used '''
        self.use_header = use_header
        if use_header:
            self.features = nn.BatchNorm1d(meta_dim, eps=1e-05)
            nn.init.constant_(self.features.weight, 1.0)
            self.features.weight.requires_grad = False
            self.id_header = FaceTransformerHeader('arcface', num_ids, 0.5)

        self.use_celebs = use_celebs
        self.use_celeb_basis = use_celeb_basis
        if use_celebs and use_celeb_basis:
            ''' which kind of normalization is the best? '''
            # self.to_weight = nn.Softmax(dim=-1)  # bias? only one big
            self.to_weight = VectorNorm(dim=-1, p=2)  # super sphere space is the best choice!
            # self.to_weight = VectorSumAs(norm_shape=inner_dim, dim=1, s=1.)  # or with weight decay?
            # self.to_weight = nn.Identity()
            # self.to_weight = nn.LayerNorm(inner_dim, eps=1e-5)  # better L2

        self.use_rm_mlp = use_rm_mlp
        if use_rm_mlp:  # for ablation study
            self.coef = nn.Parameter(torch.randn(num_ids, self.num_es, self.heads, inner_dim))

    def _vanilla_forward(self, img: torch.Tensor, id_idx: torch.Tensor):
        raise ValueError('Warning of meta_net')
        with torch.no_grad():
            img = img.permute(0, 3, 1, 2)  # to (N,C,H,W)

            # M = self.trans_matrix.repeat(img.size()[0], 1, 1)
            # print('dev', img.shape, M.shape)
            # img = kornia.geometry.transform.warp_affine(img, M, (512, 512))

            M = self.trans_matrix.repeat(img.size()[0], 1, 1)  # to (B,2,3)
            grid = F.affine_grid(M, size=img.size(), align_corners=True)  # 得到grid 用于grid sample
            img = F.grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine

            img = F.interpolate(img, size=112, mode="bilinear", align_corners=True)

            # print('[input img]', img[0, 0, 40:80, 40:80])
            # print('loaded', self.id_model.conv1.weight)

            # from PIL import Image
            # from torchvision.transforms import transforms
            # id1 = Image.open("infer_images/training/00000_id0_#0.jpg").convert("RGB")
            # id2 = Image.open("infer_images/training/00088_id8_#8.jpg").convert("RGB")
            # trans = transforms.Compose([
            #     transforms.Resize(112),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])
            #
            # def pil_to_4d(img_pil: Image):
            #     tensor = trans(img_pil).permute(1, 2, 0)
            #     tensor = tensor.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)  # (N,H,W,C)
            #     return tensor.to(img.device)
            #
            # faces = torch.cat([pil_to_4d(id1), pil_to_4d(id2)], dim=-1)
            # faces = faces.permute(0, 3, 1, 2)  # (N,C,H,W)
            # x = self.id_model(faces[:, :3, :, :])
            # print('[id_vec1]', x[0, 200:300])
            # x = self.id_model(faces[:, 3:, :, :])
            # print('[id_vec2]', x[0, 200:300])
            # return img

            self.id_model.eval()
            x = self.id_model(img)
            # print('[id_vec2]', x[0, 200:300])
            # return x
            x = F.normalize(x, dim=-1, p=2)

        # x = self.common_mlp(x)
        # x = self.common_mlp2(x)
        x = self.stylegan_mlp(x)

        if self.use_expert:
            raise ValueError('Expert not supported now.')
            b = id_idx.shape[0]
            z = torch.zeros(b, self.meta_dim).to(img.device)
            for b_idx in range(b):
                z[b_idx] = self.experts[int(id_idx[b_idx])](x[b_idx])
        else:
            z = x

        if self.use_header:
            with torch.no_grad():
                z = self.features(z)
            if not self.training:
                return z, None
            else:
                pred_cls = self.id_header(z, id_idx)
                return z, pred_cls

        z = F.normalize(z, dim=1)

        return z, None
    
    def _celebs_forward(self, img: torch.Tensor, id_idx: torch.Tensor,
                        celebs_embeds: torch.Tensor, aligned_faces=None, h_space=None, t=None, w_latents=None, delta_w=None):
        if not self.use_rm_mlp:
            with torch.no_grad():
                img = img.permute(0, 3, 1, 2)  # to (N,C,H,W)

                if(w_latents is None and self.domain_name=="face"):
                    # print("w_latents is None")
                    if(aligned_faces is not None):
                        img = transforms.Resize((256, 256))(aligned_faces.permute(0, 3, 1, 2))
                    else:
                        img = transforms.Resize((256, 256))(img)
                    w_latents = self.get_wlatents(img)
                elif(w_latents is None and self.domain_name in ["human", "dress"]):
                    if(aligned_faces is not None):
                        w_latents = self.get_human_e4e_inversion(aligned_faces.permute(0,3,1,2))
                    else:
                        w_latents = self.get_human_e4e_inversion(img)   
                else:
                    w_latents = w_latents.to(img.device)

                if(delta_w is not None):
                    w_latents = w_latents + delta_w  # size [b, 18, 512]


            # TODO add hspace and find suitable model
            if(self.use_stylegan_based_mapper):
                x = self.stylegan_mlp(h_space, w_latents, t)
                x = x.reshape(w_latents.shape[0], -1)
                if(self.norm_reg_embedding is not None):
                    x = x[:, :self.num_es * self.context_dim]
                    x_pass = x[:, self.num_es * self.context_dim:]
            else:
                x = self.stylegan_mlp(w_latents)
            
            x = rearrange(x, 'b (e d) -> b e d',
                          e=self.num_es).contiguous()  # (N,num_es,heads,inner_dim)
            
            if(self.domain_clip_embed is not None and not self.use_basis_offset):
                assert self.use_celeb_basis == False
                alpha = 0.1
                reg_loss = torch.mean(torch.norm(x, dim=-1))
                x = self.domain_clip_embed.reshape(self.num_es, -1).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device) + alpha * x
            else:
                reg_loss = 0
            
            if(self.norm_reg_embedding is not None):
                assert self.use_celeb_basis == False
                x = (x/(torch.norm(x, dim=2, keepdim=True) + 1e-5)) * self.norm_reg_embedding.to(x.device)

            x = x.unsqueeze(2)
            if celebs_embeds is not None and self.use_celeb_basis:
                assert self.use_celebs == True
                if(not self.use_celeb_basis):
                    x = self.to_weight(x)  # (N,num_es,heads,inner_dim)  # Normalising the mapper output

                if(self.shift_basis is not None):
                    x = x * torch.exp(self.scale_factor.to(x.device)) + self.shift_factor.to(x.device)


        if celebs_embeds is not None and self.use_celeb_basis:
            ''' 3dmm/pca-based svd '''
            c_all = celebs_embeds.to(x.device).detach()  # (es,1+inner_dim,768)
            c_mean, pca_base = c_all[:, 0], c_all[:, 1:]  # mean:(es,768), pca_base:(es,inner_dim,768)

            if(self.use_basis_offset):
                pca_base = self.basis_offset.to(x.device)
                pca_base = normalize_basis_components(pca_base)
                c_mean = self.domain_clip_embed.reshape(2, -1).to(x.device)
                orthonormality_loss = torch.mean(torch.norm(torch.matmul(pca_base, pca_base.transpose(1, 2)) 
                                                                   - torch.eye(pca_base.shape[1]).to(pca_base.device), dim=(1, 2)))
                reg_loss = reg_loss  + orthonormality_loss
                print("[orthonormality_loss]", orthonormality_loss)
                
            c_mean = c_mean.unsqueeze(1).unsqueeze(0)  # mean:(1,es,1,768)
            if not self.vis_mean:
                # z = torch.einsum('b e h k, e k c -> b e h c', x, pca_base) + c_mean  # (N,num_es,heads,768)
                z = torch.einsum('b e h k, e k c -> b e h c', x, pca_base) 
                if(self.use_regularization):
                    reg_loss += 1e-2 * torch.mean(torch.norm(z, dim=-1))
                    z = c_mean + 0.1 * z
                else:
                    z = c_mean + z
            else:   # for ablation study
                mean, std = self.vis_mean_params if self.vis_mean_params is not None else (0., 0.)
                noise = torch.randn_like(x, device=x.device) * std + mean
                z = torch.einsum('b e h k, e k c -> b e h c', noise, pca_base) + c_mean  # (N,num_es,heads,768)
        else:
            z = x
            assert x.shape[-1] == self.context_dim

        z = rearrange(z, 'b e h c -> b (e h) c').contiguous()  # (N,num_es*heads,768)

        ''' for calculating id loss (not used) '''
        # if self.use_header:  # num_es not supported here
        #     with torch.no_grad():
        #         z = self.features(z)
        #     if not self.training:
        #         return z, None, x
        #     else:
        #         pred_cls = self.id_header(z, id_idx)
        #         return z, pred_cls, None

        return z, None, None, reg_loss

    def forward(self, img, id_idx, celebs_embeds=None, aligned_faces=None, h_space=None, t=None, w_latents=None, delta_w=None):
        if self.use_celebs:
            return self._celebs_forward(img, id_idx, celebs_embeds, aligned_faces, h_space, t, w_latents, delta_w)
        else:
            return self._vanilla_forward(img, id_idx)  # deprecated

    def forward_double_faces(self, img_dual: torch.Tensor, id_dual: torch.Tensor):
        """
        :param img_dual: (N,H,W,2C)
        :param id_dual: (N,2)
        """
        vec1 = self.forward(img_dual[:, :, :, :3], id_dual[:, 0])
        vec2 = self.forward(img_dual[:, :, :, 3:], id_dual[:, 1])
        return vec1, vec2

    def forward_triple_faces(self, img_triple: torch.Tensor, id_triple: torch.Tensor):
        """
        :param img_triple: (N,H,W,3C)
        :param id_triple: (N,3)
        """
        vec1, cls1 = self.forward(img_triple[:, :, :, :3], id_triple[:, 0])
        vec2, cls2 = self.forward(img_triple[:, :, :, 3:6], id_triple[:, 1])
        vec3, cls3 = self.forward(img_triple[:, :, :, 6:], id_triple[:, 2])
        return vec1, vec2, vec3, cls1, cls2, cls3

    def forward_multi_faces(self, img_multi: torch.Tensor, id_multi: torch.Tensor, celeb_embeds=None, aligned_faces=None,
                             h_space=None, w_latents=None, delta_w=None):
        """
        :param img_multi: (N,H,W,(1+diff+1+diff)C)
        :param id_multi: (N,1+diff+1+diff)
        :param celeb_embeds: (es,1+inner_dim,768)
        """
        b, num = id_multi.shape
        chunked = img_multi.chunk(num, -1)
        concatenated = torch.cat(chunked, 0)  # ((1+diff+1+diff)N,H,W,C), num=1+diff+1+diff

        if(aligned_faces is not None):
            align_chunked = aligned_faces.chunk(num, -1)
            align_concatenated = torch.cat(align_chunked, 0)
        else:
            align_concatenated = None
        if(h_space is not None):
            h_space_feat = h_space['h_space_feat']
            # For new dataset class not needed
            # h_space_chunked = h_space_feat.chunk(num, -1)
            # h_space_concatenated = torch.cat(h_space_chunked, 0)

            h_space_concatenated = h_space_feat
            ts = h_space['t'] if num == 1 else torch.cat([h_space['t']]*num, dim=0)
        else:
            h_space_concatenated = None
            ts = h_space['t'] if num == 1 else torch.cat([h_space['t']]*num, dim=0)
        # print("[b, num, id_multi, chunked, concatenated]", b, num, id_multi.shape, len(chunked), concatenated.shape)
        vec, cls, cef, reg_loss = self.forward(concatenated,
                                        torch.flatten(id_multi),
                                        celeb_embeds, align_concatenated, h_space_concatenated, ts, w_latents, delta_w)
            
        # vec:(num*N,es*h,768), cls:(num*N,es,cls)('h' not supported), cef:(num*N,es,h,inner_dim)
        vec = vec.chunk(num, 0)  # num*(N,es*h,768)
        if cls is not None:
            cls = cls.chunk(num, 0)  # num*(N,es,cls)
        if(cef is not None):
            cef = cef.chunk(num, 0)  # num*(N,es,h,inner_dim)
        return vec, cls, cef, reg_loss

    def load_fr_net(self):
        self.id_model = iresnet100()
        id_path = './weights/glint360k_cosface_r100_fp16_0.1/backbone.pth'
        weights = torch.load(id_path)
        self.id_model.load_state_dict(weights)
        for param in self.id_model.parameters():
            param.requires_grad = False
        self.id_model.eval()
    
    def load_e4e_with_decoder(self,checkpoint_path, device='cuda'):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts["test_batch_size"] = 1
        # print(opts)

        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = device
        opts = argparse.Namespace(**opts)

        self.e4e_encoder = pSp(opts)
        self.e4e_encoder.eval()
        self.e4e_encoder.cuda()
        for param in self.e4e_encoder.parameters():
            param.requires_grad = False
        latent_avg = ckpt['latent_avg'].to(device)

    def get_wlatents(self, x, is_cars=False):
        codes = self.e4e_encoder.encoder(x)
        self.e4e_encoder.latent_avg = self.e4e_encoder.latent_avg.to(codes.device)
        if self.e4e_encoder.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.e4e_encoder.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
        return codes

    def get_human_e4e_inversion(self, image):
        # image = (image + 1) / 2
        new_image = self.e4e_image_transform(image).to(image.device)
        _, w = self.e4e_encoder(new_image, randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        # if self.use_wandb:
            # log_image_from_w(w, self.G, 'First e4e inversion')
        return w
    
    def trainable_state_dict(self, verbose=False):
        trainable_state_dict = {}
        for key, val in self.state_dict().items():
            if 'stylegan_mlp' in key:
                trainable_state_dict[key] = val
            # adding new mapper networks
            if 'final_mlp' in key:
                trainable_state_dict[key] = val
            if 'hspace_cnn' in key:
                trainable_state_dict[key] = val
            if 'basis_offset' in key:
                trainable_state_dict[key] = val
            if 'scale_factor' in key:
                trainable_state_dict[key] = val
            if 'shift_factor' in key:
                trainable_state_dict[key] = val
            
            if self.use_rm_mlp and 'coef' in key:
                trainable_state_dict[key] = val
        if verbose:
            # print('[meta_net] trainable_state_dict ready to save.', list(trainable_state_dict.keys()))
            pass
        return trainable_state_dict

    def load_trainable_state_dict(self, state_dict: dict, verbose=False):
        trainable_state_dict = {}
        for key, val in state_dict.items():
            if 'stylegan_mlp' in key:
                trainable_state_dict[key] = val
            # adding new mapper networks
            if 'final_mlp' in key:
                trainable_state_dict[key] = val
            if 'hspace_cnn' in key:
                trainable_state_dict[key] = val   
            if 'basis_offset' in key:
                trainable_state_dict[key] = val
            if 'scale_factor' in key:
                trainable_state_dict[key] = val
            if 'shift_factor' in key:
                trainable_state_dict[key] = val

            if self.use_rm_mlp and 'coef' in key:
                trainable_state_dict[key] = val
        self.load_state_dict(trainable_state_dict, strict=False)
        if verbose:
            # print('[meta_net] trainable_state_dict loaded.', list(trainable_state_dict.keys()))
            pass

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FaceTransformerHeader(nn.Module):
    def __init__(self,
                 header_type: str,
                 header_num_classes: int,
                 header_params_m: float,
                 header_params_s: float = 64.0,
                 header_params_a: float = 0.,
                 header_params_k: float = 0.,
                 ):
        super(FaceTransformerHeader, self).__init__()
        feature_dim = 768
        from ldm.modules.id_embedding.margin_losses import AMCosFace, Softmax, AMArcFace
        header_type = header_type.lower()
        if 'cosface' in header_type:
            self.loss = AMCosFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'arcface' in header_type:
            self.loss = AMArcFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'softmax' in header_type:
            self.loss = Softmax(in_features=feature_dim,
                                out_features=header_num_classes,
                                device_id=None, )
        else:
            raise ValueError('Header type not supported.')

    def forward(self, v, label=None):
        if self.training:
            final = self.loss(v, label)
            return final  # id:(b, dim)
        else:
            return v


if __name__ == "__main__":
    my_net = MetaIdNet(use_expert=False)
    for _ in range(1):
        face_img = torch.ones((2, 6, 512, 512), dtype=torch.float32).permute(0, 2, 3, 1) * 10  # (N,H,W,C)
        face_img[:, :, :, :3] += 1
        face_ids = torch.randint(10, size=(2,)).unsqueeze(0).repeat(2, 1)
        embed1, embed2 = my_net.forward_double_faces(face_img, face_ids)
        print('in_ids:', face_ids)
        print('embed.shape', embed1.shape)
        print('embed1:', embed1)
        print('embed2:', embed2)
    embed1.mean().backward()
    print('Backward check ok.')
