import torch
import torch.nn as nn
from einops import rearrange, einsum
from omegaconf import DictConfig

from models.base import Model
from models.modules import TimestepEmbedder, CrossAttentionLayer, SelfAttentionBlock
from models.scene_models.pointtransformer import TransitionDown, TransitionUp, PointTransformerBlock
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from models.functions import load_scene_model

class PointSceneMLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, widening_factor: int=1, bias: bool=True) -> None:
        super().__init__()

        self.mlp_pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, widening_factor * in_dim, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * in_dim, out_dim, bias=bias),
        )

        out_dim = out_dim * 2
        self.mlp_post = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim, bias=bias),
            nn.GELU(),
            nn.Linear(out_dim, out_dim // 2, bias=bias),
        )

    def forward(self, point_feat: torch.Tensor) -> torch.Tensor:
        point_feat = self.mlp_pre(point_feat)
        scene_feat = point_feat.mean(dim=1, keepdim=True).repeat(1, point_feat.shape[1], 1)
        point_feat = torch.cat([point_feat, scene_feat], dim=-1)
        point_feat = self.mlp_post(point_feat)

        return point_feat

class ContactMLP(nn.Module):

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        self.point_mlp_dims = arch_cfg.point_mlp_dims
        self.point_mlp_widening_factor = arch_cfg.point_mlp_widening_factor
        self.point_mlp_bias = arch_cfg.point_mlp_bias

        layers = []
        idim = contact_dim + point_feat_dim + text_feat_dim + time_emb_dim
        for odim in self.point_mlp_dims:
            layers.append(PointSceneMLP(idim, odim, widening_factor=self.point_mlp_widening_factor, bias=self.point_mlp_bias))
            idim = odim
        self.point_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        if point_feat is not None:
            bs, num_points, point_feat_dim = point_feat.shape
            x = torch.cat([
                x,
                point_feat,
                language_feat.repeat(1, num_points, 1),
                time_embedding.repeat(1, num_points, 1)
            ], dim=-1) # [bs, num_points, contact_dim + point_feat_dim + language_feat_dim + time_embedding_dim]
        else:
            x = torch.cat([
                x,
                language_feat.repeat(1, num_points, 1),
                time_embedding.repeat(1, num_points, 1)
            ], dim=-1) # [bs, num_points, contact_dim + language_feat_dim + time_embedding_dim]
        x = self.point_mlp(x) # [bs, num_points, point_mlp_dim[-1]]

        return x


class ContactPerceiver(nn.Module):
    
    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        self.point_pos_emb = arch_cfg.point_pos_emb

        self.encoder_q_input_channels = arch_cfg.encoder_q_input_channels
        self.encoder_kv_input_channels = arch_cfg.encoder_kv_input_channels
        self.encoder_num_heads = arch_cfg.encoder_num_heads
        self.encoder_widening_factor = arch_cfg.encoder_widening_factor
        self.encoder_dropout = arch_cfg.encoder_dropout
        self.encoder_residual_dropout = arch_cfg.encoder_residual_dropout
        self.encoder_self_attn_num_layers = arch_cfg.encoder_self_attn_num_layers
        
        self.decoder_q_input_channels = arch_cfg.decoder_q_input_channels
        self.decoder_kv_input_channels = arch_cfg.decoder_kv_input_channels
        self.decoder_num_heads = arch_cfg.decoder_num_heads
        self.decoder_widening_factor = arch_cfg.decoder_widening_factor
        self.decoder_dropout = arch_cfg.decoder_dropout
        self.decoder_residual_dropout = arch_cfg.decoder_residual_dropout

        self.language_adapter = nn.Linear(
            text_feat_dim,
            self.encoder_q_input_channels,
            bias=True)
        self.time_embedding_adapter = nn.Linear(
            time_emb_dim,
            self.encoder_q_input_channels,
            bias=True)

        self.encoder_adapter = nn.Linear(
            contact_dim + point_feat_dim + (3 if self.point_pos_emb else 0), 
            self.encoder_kv_input_channels,
            bias=True)
        self.decoder_adapter = nn.Linear(
            self.encoder_kv_input_channels,
            self.decoder_q_input_channels,
            bias=True)

        self.encoder_cross_attn = CrossAttentionLayer(
            num_heads=self.encoder_num_heads,
            num_q_input_channels=self.encoder_q_input_channels,
            num_kv_input_channels=self.encoder_kv_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=self.encoder_self_attn_num_layers,
            num_heads=self.encoder_num_heads,
            num_channels=self.encoder_q_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.decoder_cross_attn = CrossAttentionLayer(
            num_heads=self.decoder_num_heads,
            num_q_input_channels=self.decoder_q_input_channels,
            num_kv_input_channels=self.decoder_kv_input_channels,
            widening_factor=self.decoder_widening_factor,
            dropout=self.decoder_dropout,
            residual_dropout=self.decoder_residual_dropout,
        )

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        if point_feat is not None:
            x = torch.cat([x, point_feat], dim=-1) # [bs, num_points, contact_dim + point_feat_dim]
        if self.point_pos_emb:
            point_pos = kwargs['c_pc_xyz']
            x = torch.cat([x, point_pos], dim=-1) # [bs, num_points, contact_dim + point_feat_dim + 3]

        # encoder
        enc_kv = self.encoder_adapter(x) # [bs, num_points, enc_kv_dim]

        language_feat = self.language_adapter(language_feat) # [bs, 1, enc_q_dim]
        time_embedding = self.time_embedding_adapter(time_embedding) # [bs, 1, enc_q_dim]
        enc_q = torch.cat([language_feat, time_embedding], dim=1) # [bs, 1 + 1, enc_q_dim]

        enc_q = self.encoder_cross_attn(enc_q, enc_kv).last_hidden_state
        enc_q = self.encoder_self_attn(enc_q).last_hidden_state

        # decoder
        dec_kv = enc_q
        dec_q = self.decoder_adapter(enc_kv) # [bs, num_points, dec_q_dim]
        dec_q = self.decoder_cross_attn(dec_q, dec_kv).last_hidden_state # [bs, num_points, dec_q_dim]

        return dec_q

class ContactPointTrans(nn.Module):

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        self.num_points = arch_cfg.num_points

        self.c = contact_dim + point_feat_dim + 3 # 3 for xyz
        block = PointTransformerBlock
        blocks = arch_cfg.blocks

        self.in_planes, planes = self.c, [64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64

        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], is_head=True)  # transform p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1

        self.ctx  = self._make_ctx(planes[3] + text_feat_dim + time_emb_dim, planes[3])

    @property
    def num_groups(self):
        return self.num_points // 64

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_ctx(self, in_planes, planes):
        layers = [
            nn.Linear(in_planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, planes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        p = kwargs['c_pc_xyz']

        if point_feat is not None:
            x = torch.cat([x, point_feat], dim=-1) # [bs, num_points, contact_dim + point_feat_dim]
        context = torch.cat([language_feat, time_embedding], dim=-1) # [bs, language_feat_dim + time_embedding_dim]

        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.IntTensor(offset).to(p0.device)
        
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        x4 = rearrange(x4, '(b n) d -> b n d', b=len(offset), n=self.num_groups)
        context = context.repeat(1, self.num_groups, 1)
        x4 = rearrange(torch.cat((x4, context), dim=-1), 'b n d -> (b n) d')
        x4 = self.ctx(x4)

        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return rearrange(x1, '(b n) d -> b n d', b=len(offset), n=offset[0]) # (b, n, planes[0])

class ContactPointTransV2(nn.Module):

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        self.num_points = arch_cfg.num_points

        self.c = contact_dim + point_feat_dim + 3 # 3 for xyz
        block = PointTransformerBlock
        blocks = arch_cfg.blocks

        self.in_planes, planes = self.c, [64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64

        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], is_head=True)  # transform p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1

        self.ctx4  = self._make_ctx(planes[3] + text_feat_dim + time_emb_dim, planes[3])
        self.ctx3  = self._make_ctx(planes[2] + text_feat_dim + time_emb_dim, planes[2])
        self.ctx2  = self._make_ctx(planes[1] + text_feat_dim + time_emb_dim, planes[1])

        self.self_attn_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=planes[-1],
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu',
                batch_first=True,
            ),
            num_layers=1
        )
    
    @property
    def num_groups(self):
        return self.num_points // 64

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_ctx(self, in_planes, planes):
        layers = [
            nn.Linear(in_planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Linear(planes, planes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        p = kwargs['c_pc_xyz']

        if point_feat is not None:
            x = torch.cat([x, point_feat], dim=-1) # [bs, num_points, contact_dim + point_feat_dim]
        context = torch.cat([language_feat, time_embedding], dim=-1) # [bs, 1, language_feat_dim + time_embedding_dim]
        
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.IntTensor(offset).to(p0.device)
        
        ## transition down
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        ## transition up
        x4 = rearrange(x4, '(b n) d -> b n d', b=len(offset))
        x4 = self.self_attn_layers(x4)
        x4 = rearrange(torch.cat((x4, context.repeat(1, x4.shape[1], 1)), dim=-1), 'b n d -> (b n) d')
        x4 = self.ctx4(x4)
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4]), o4])[1]

        x3 = rearrange(x3, '(b n) d -> b n d', b=len(offset))
        x3 = rearrange(torch.cat((x3, context.repeat(1, x3.shape[1], 1)), dim=-1), 'b n d -> (b n) d')
        x3 = self.ctx3(x3)
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]

        x2 = rearrange(x2, '(b n) d -> b n d', b=len(offset))
        x2 = rearrange(torch.cat((x2, context.repeat(1, x2.shape[1], 1)), dim=-1), 'b n d -> (b n) d')
        x2 = self.ctx2(x2)
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]

        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return rearrange(x1, '(b n) d -> b n d', b=len(offset)) # (b, n, planes[0])

@Model.register()
class CDM(nn.Module):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.contact_type = cfg.data_repr
        self.contact_dim = cfg.input_feats

        ## time embedding
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.time_emb_dim, self.time_emb_dim, max_len=1000)

        ## text
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError

        ## scene
        if not cfg.scene_model.use_scene_model:
            self.point_feat_dim = 0
        elif cfg.scene_model.use_openscene:
            self.point_feat_dim = cfg.scene_model.point_feat_dim
        else:
            self.scene_model_dim = 3 + int(cfg.scene_model.use_color) * 3
            self.freeze_scene_model = cfg.scene_model.freeze
            self.scene_model = load_scene_model(
                cfg.scene_model.name, self.scene_model_dim, cfg.scene_model.num_points, cfg.scene_model.pretrained_weight, freeze=self.freeze_scene_model)
            self.point_feat_dim = cfg.scene_model.point_feat_dim

        ## model architecture
        self.arch = cfg.arch
        if self.arch == 'MLP':
            self.arch_cfg = cfg.arch_mlp
            CONTACT_MODEL = ContactMLP
        elif self.arch == 'Perceiver':
            self.arch_cfg = cfg.arch_perceiver
            CONTACT_MODEL = ContactPerceiver
        elif self.arch == 'PointTrans':
            self.arch_cfg = cfg.arch_pointtrans
            CONTACT_MODEL = ContactPointTrans
        elif self.arch == 'PointTransV2':
            self.arch_cfg = cfg.arch_pointtrans
            CONTACT_MODEL = ContactPointTransV2
        else:
            raise NotImplementedError
        self.contact_model = CONTACT_MODEL(
            self.arch_cfg,
            contact_dim=self.contact_dim,
            point_feat_dim=self.point_feat_dim,
            text_feat_dim=self.text_feat_dim,
            time_emb_dim=self.time_emb_dim
        )

        self.contact_layer = nn.Linear(self.arch_cfg.last_dim, self.contact_dim, bias=True)

    def forward(self, x, timesteps, **kwargs):
        """ Forward pass of the model.
        
        Args:
            x: input contact map, [bs, num_points, contact_dim]
            kwargs: other inputs, e.g., text, etc.
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        ## time embedding
        time_emb = self.timestep_embedder(timesteps) # [bs, 1, time_emb_dim]

        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
        elif self.text_feat_type == 'bert':
            text_emb = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'], max_length=self.text_max_length, s_feat=True, device=self.device)
        else:
            raise NotImplementedError
        text_emb = text_emb.unsqueeze(1).detach().float()  # [bs, 1, text_feat_dim]

        ## scene embedding
        if not hasattr(self, 'scene_model'):
            if self.point_feat_dim == 0:
                pc_emb = None
            elif self.point_feat_dim == 1:
                if kwargs['c_pc_feat'].shape[-1] == 1:
                    pc_emb = kwargs['c_pc_feat']
                else:
                    pc_emb = einsum(kwargs['c_pc_feat'], text_emb, 'b n d, b m d -> b n m') # [bs, num_points, 1]
            else:
                pc_emb = kwargs['c_pc_feat'] # [bs, num_points, 768]
        else:
            pc_emb = self.scene_model((kwargs['c_pc_xyz'], kwargs['c_pc_feat'])).detach() # [bs, num_points, point_feat_dim]

        x = self.contact_model(x, pc_emb, text_emb, time_emb, **kwargs) # [bs, num_points, last_dim]
        x = self.contact_layer(x) # [bs, num_points, contact_dim]

        return x
