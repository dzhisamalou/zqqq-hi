import math
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch as th # 添加 torch 别名以匹配注释
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.sgm.modules.attention import SpatialTransformer
from models.sgm.modules.diffusionmodules.util import (
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from models.sgm.modules.video_attention import (
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    MemoryEfficientCrossAttention,
)
from models.sgm.util import default
from models.sgm.modules.diffusionmodules.model import ResBlock, Downsample, TimestepEmbedSequential, Upsample, Timestep


class VideoResBlock(ResBlock):
    """
    视频残差块，继承自基础ResBlock，增加了对时间维度的处理能力
    用于视频生成模型中处理时间一致性
    """
    def __init__(
        self,
        channels: int,                              # 输入通道数
        emb_channels: int,                          # 时间嵌入的通道数
        dropout: float,                             # dropout比率
        video_kernel_size: Union[int, List[int]] = 3, # 视频卷积核大小
        merge_strategy: str = "fixed",              # 融合策略："fixed"(固定),"learned"(学习)等
        merge_factor: float = 0.5,                  # 融合因子，控制空间和时间特征的混合比例
        out_channels: Optional[int] = None,         # 输出通道数
        use_conv: bool = False,                     # 是否使用卷积进行通道变换
        use_scale_shift_norm: bool = False,         # 是否使用缩放偏移归一化
        dims: int = 2,                              # 空间维度数
        use_checkpoint: bool = False,               # 是否使用梯度检查点来节省内存
        up: bool = False,                           # 是否上采样
        down: bool = False,                         # 是否下采样
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )

        # 时间堆叠模块，处理视频帧之间的时间关系
        self.time_stack = ResBlock(
            default(out_channels, channels),        # 如果out_channels为None则使用channels
            emb_channels,
            dropout=dropout,
            dims=3,                                 # 3维卷积处理时空信息
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,                # 交换时间嵌入维度
        )
        # 时间混合器，用于混合空间特征和时间特征
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,                     # 混合因子
            merge_strategy=merge_strategy,          # 混合策略
            rearrange_pattern="b t -> b 1 t 1 1",   # 重排模式，调整张量形状
        )

    def forward(
        self,
        x: th.Tensor,                              # 输入特征 [B*T, C, H, W]，实际为[16, C, H, W]，因为B=1,T=16
        emb: th.Tensor,                            # 时间嵌入 [B*T, emb_C]
        num_video_frames: int,                     # 视频帧数，这里为16
        image_only_indicator: Optional[th.Tensor] = None, # 只有图像的指示器
    ) -> th.Tensor:
        # 首先通过基类的forward处理
        x = super().forward(x, emb)

        # 将特征从[B*T, C, H, W]重排为[B, C, T, H, W]，以便进行时间处理
        # 对于B=1,T=16的情况，这变成[1, C, 16, H, W], H, W]
        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        # 应用时间堆叠处理时间维度上的特征
        x = self.time_stack(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        )
        # 混合空间和时间特征
        x = self.time_mixer(
            x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
        )
        # 将特征从[B, C, T, H, W]重排回[B*T, C, H, W]，即[16, C, H, W]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class VideoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        flow_channels: Optional[int] = None, # flow 特征通道数
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.flow_channels = flow_channels # 存储 flow 通道数

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        # 新增: 定义 flow 特征投影层
        if self.flow_channels is not None:
            self.flow_proj = conv_nd(dims, self.flow_channels, self.in_channels, 1)
        else:
            self.flow_proj = None

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: th.Tensor,                              # 输入: [B*T, C, H, W]，实际为[16, C, H, W]
        timesteps: th.Tensor,                      # 时间步: [B*T]，实际为[16]
        context: Optional[th.Tensor] = None,       # 上下文条件，现在可能是[B*T, 49, 1024]形状的DIL tokens
        y: Optional[th.Tensor] = None,             # 类别标签
        time_context: Optional[th.Tensor] = None,  # 时间上下文
        num_video_frames: Optional[int] = 16,      # 视频帧数，默认设为16
        image_only_indicator: Optional[th.Tensor] = None, # 图像指示器
        flow_features: Optional[th.Tensor] = None, # 低层属性特征 flow
        frame_mask: Optional[th.Tensor] = None,    # 控制 flow 注入的掩码 Mf
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"

        # 注入 flow 特征到输入 x
        if flow_features is not None and self.flow_proj is not None:
            projected_flow = self.flow_proj(flow_features)
            if frame_mask is not None:
                # 确保 frame_mask 可以广播
                # 假设 frame_mask 形状为 [B*T] 或 [B*T, 1, 1, 1]
                if frame_mask.ndim == 1:
                    frame_mask = rearrange(frame_mask, 'bt -> bt 1 1 1')
                projected_flow = projected_flow * frame_mask
            x = x + projected_flow

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # 21 x 320
        emb = self.time_embed(t_emb) # 21 x 1280

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y) # 21 x 1280

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,  # 这里context是扩展后的DIL tokens [B*T, 49, 1024]
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,  # 传递帧数16
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,  
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context, 
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)
