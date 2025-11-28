# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMARL model. Adapted from HF LLaMa model."""
import functools
import inspect
import math
from typing import Callable, List, Optional, Tuple, Union, Dict
import warnings
import numpy as np

import torch
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func

from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaMLP,
    apply_rotary_pos_emb,
    repeat_kv,
    _make_causal_mask,
)
from transformers.utils import logging
from dataclasses import dataclass
from .configuration_llamarl import LlamaRLConfig

logger = logging.get_logger(__name__)

@dataclass
class ACOutputWithPast(BaseModelOutputWithPast):                ## use this now
    softprompt: Optional[torch.FloatTensor] = None
    sampled_seg_length: Optional[int] = None

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(2 * self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        emb_cos = emb.cos()
        emb_sin = emb.sin()
        self.register_buffer("cos_cached", emb_cos[self.max_seq_len_cached:].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb_sin[self.max_seq_len_cached:].to(dtype), persistent=False)

        l2rcumsum_cos = emb_cos[:self.max_seq_len_cached] + emb_cos[:self.max_seq_len_cached].sum(dim=0, keepdim=True) - emb_cos[:self.max_seq_len_cached].cumsum(dim=0)
        l2rcumsum_sin = emb_sin[:self.max_seq_len_cached] + emb_sin[:self.max_seq_len_cached].sum(dim=0, keepdim=True) - emb_sin[:self.max_seq_len_cached].cumsum(dim=0)
        self.register_buffer("cos_cumsum_cached", l2rcumsum_cos.to(dtype), persistent=False)
        self.register_buffer("sin_cumsum_cached", l2rcumsum_sin.to(dtype), persistent=False)

    def forward(self, x, seq_len=None, cumsum=False):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        if cumsum:
            return (
                self.cos_cumsum_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cumsum_cached[:seq_len].to(dtype=x.dtype),
            )
        else:
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype),
            )



def create_episodes_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    eps_indexes = (1 - mask.float()).cumsum(dim=1)
    expanded_attn_mask = eps_indexes[:, None] == eps_indexes[:, -tgt_len:, None]
    expanded_attn_mask = expanded_attn_mask[:, None].to(dtype)

    inverted_mask = 1.0 - expanded_attn_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

# from https://github.com/opendilab/DI-engine/blob/6137b5e9ae7947186209bf54d998b922156aa360/ding/torch_utils/network/gtrxl.py#L44
class GRUGatingUnit(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Arguments:
            - input_dim: (:obj:`int`): dimension of input.
            - bg (:obj:`bg`): gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """
        super(GRUGatingUnit, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute output value with gating mechanism
        Arguments:
            - x: (:obj:`torch.Tensor`): first input.
            - y: (:obj:`torch.Tensor`): second input.
            x and y have same shape and last shape is input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): output of GRU. Same shape of x and y.
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape


class LlamaRLAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaRLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_casual = config.is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.position_embed_type = config.position_embed_type
        self.sequence_embed_type = config.sequence_embed_type
        if self.position_embed_type == "rope" or config.sequence_embed_type == "rope":
            self._init_rope()

        self.mul_factor_for_sink_attn = config.mul_factor_for_sink_attn
        self.add_sink_kv = config.add_sink_kv
        self.add_sink_tokens = config.add_sink_tokens
        self.num_sink_tokens = config.num_sink_tokens
        if self.add_sink_kv:
            self.add_sink_tokens = True

        if not self.add_sink_tokens:
            self.num_sink_tokens = 0

        if self.add_sink_kv:
            self.head_bias_key = torch.nn.Parameter(data=torch.zeros(1, self.num_key_value_heads, 1, self.head_dim), requires_grad=config.is_sink_k_trainable)
            self.head_bias_value = torch.nn.Parameter(data=torch.zeros(1, self.num_key_value_heads, 1, self.head_dim), requires_grad=config.is_sink_v_trainable)


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=self.max_position_embeddings)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([key_states[:, :, :self.num_sink_tokens], past_key_value[0], key_states[:, :, self.num_sink_tokens:]], dim=2)
            value_states = torch.cat([value_states[:, :, :self.num_sink_tokens], past_key_value[1], value_states[:, :, self.num_sink_tokens:]], dim=2)
        past_key_value = (key_states[:, :, self.num_sink_tokens:], value_states[:, :, self.num_sink_tokens:]) if use_cache else None

        if self.add_sink_kv:
            head_bias_key = self.head_bias_key.to(key_states.dtype).expand(bsz, -1, -1, -1)
            head_bias_value = self.head_bias_value.to(key_states.dtype).expand(bsz, -1, -1, -1)
            key_states[:, :, :self.num_sink_tokens] = head_bias_key
            value_states[:, :, :self.num_sink_tokens] = head_bias_value
            query_states[:, :, :self.num_sink_tokens] = 0

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if not output_attentions:                                                                                   ## what's happening hwerre
            if attention_mask is not None:
                attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
            else:
                query_states = query_states.transpose(1, 2).half()
                key_states = key_states.transpose(1, 2).half()
                value_states = value_states.transpose(1, 2).half()
                attn_output = flash_attn_func(query_states, key_states, value_states, causal=self.is_casual)
                attn_output = attn_output.transpose(1, 2).contiguous().float()
        else:

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaRLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.depth_dropout_p = config.depth_dropout_p
        self.self_attn = LlamaRLAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gated_residual = config.gated_residual
        if self.gated_residual:
            self.gate1 = GRUGatingUnit(self.hidden_size)
            self.gate2 = GRUGatingUnit(self.hidden_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        if self.depth_dropout_p > 0 and self.training:
            # TODO: Use config
            factor = (torch.rand(1, device=hidden_states.device) > self.depth_dropout_p).to(hidden_states.dtype)
            mul_factor = 1.0
        else:
            factor = 1.0
            mul_factor = 1.0


        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        if self.gated_residual:
            hidden_states = self.gate1(residual, hidden_states)
        else:
            hidden_states = mul_factor * (residual + factor*hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.gated_residual:
            hidden_states = self.gate2(residual, hidden_states)
        else:
            hidden_states = mul_factor * (residual + factor*hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaRLPreTrainedModel(PreTrainedModel):
    config_class = LlamaRLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaRLDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` methid
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(functools.partial(self._set_gradient_checkpointing, value=True))
            logger.warn(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` methid
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warn(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(functools.partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()


class LlamaRLModel(LlamaRLPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaRLDecoderLayer`]

    Args:
        config: LlamaRLConfig
    """

    def __init__(self, config: LlamaRLConfig):
        super().__init__(config)
        self.inter_episodes_attention = config.inter_episodes_attention
        self.reset_position_index = config.reset_position_index
        self.add_sequence_idx_embed = config.add_sequence_idx_embed
        self.context_len = config.context_len
        self.banded_attention = config.banded_attention
        self.orphan_steps_attention = config.orphan_steps_attention
        self.add_sink_kv = config.add_sink_kv
        self.add_sink_tokens = config.add_sink_tokens
        self.num_sink_tokens = config.num_sink_tokens

        assert not self.add_sink_tokens or not self.add_sink_kv
        
        if self.add_sink_kv:
            self.add_sink_tokens = True

        if not self.add_sink_tokens:
            self.num_sink_tokens = 0
        else:
            self.sink_tokens = torch.nn.Parameter(data=torch.zeros(1, self.num_sink_tokens, config.hidden_size), requires_grad=True)


        self.position_embed_type = config.position_embed_type
        self.sequence_embed_type = config.sequence_embed_type
        assert self.position_embed_type in ["rope", "linear"] and self.sequence_embed_type in ["rope", "linear"]
        assert self.position_embed_type != "rope" or self.sequence_embed_type != "rope"
        
        if self.position_embed_type == "linear":
            self.embed_pos = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
            )
        if self.add_sequence_idx_embed and self.sequence_embed_type == "linear":
            self.embed_sequence_pos = nn.Embedding(
                config.max_num_sequence, config.hidden_size
            )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings_dim(self):                                                          ## TODO: resolve confusion
        return self.config.hidden_size
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):
        
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if not self.inter_episodes_attention:
                expanded_attn_mask = create_episodes_mask(
                    attention_mask, dtype, tgt_len=input_shape[-1]
                )

                combined_attention_mask = (
                    expanded_attn_mask
                    if combined_attention_mask is None
                    else expanded_attn_mask + combined_attention_mask
                )

            if self.banded_attention:
                banded_mask = torch.full((input_shape[0], 1, input_shape[1], input_shape[1]), torch.finfo(dtype).min)
                banded_mask = torch.triu(banded_mask, diagonal=1) + torch.tril(banded_mask, diagonal=-(self.context_len+1))
                banded_mask = banded_mask[:, :, -input_shape[-1]:].to(device)

                combined_attention_mask = (
                    banded_mask
                    if combined_attention_mask is None
                    else banded_mask + combined_attention_mask
                )

            if not self.orphan_steps_attention:
                orphan_mask = (torch.cumsum(1 - attention_mask, dim=1) == 0)[:, None, None].to(dtype)
                orphan_mask.masked_fill_(orphan_mask.to(torch.bool), torch.finfo(dtype).min)
                orphan_mask = orphan_mask[:, :, -input_shape[-1]:].to(device)

                combined_attention_mask = (
                    orphan_mask
                    if combined_attention_mask is None
                    else orphan_mask + combined_attention_mask
                )
        if self.add_sink_tokens and combined_attention_mask is not None:
            combined_attention_mask[..., :self.num_sink_tokens, :] = torch.finfo(dtype).min
            combined_attention_mask[..., :self.num_sink_tokens] = 0
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You have to specify inputs_embeds")
        elif input_ids is not None:
            raise ValueError("You have to specify inputs_embeds")
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify inputs_embeds")

        if self.add_sink_tokens:
            inputs_embeds = torch.concat(
                [
                    self.sink_tokens.expand((inputs_embeds.shape[0], -1, -1)),
                    inputs_embeds
                ], dim=1
            )
            if attention_mask is not None:
                attention_mask = torch.concat(
                    [
                        torch.zeros(attention_mask.shape[0], self.num_sink_tokens, device=attention_mask.device),
                        attention_mask
                    ], 
                    dim=1
                )

        if any(
            [
                not self.inter_episodes_attention,
                self.reset_position_index,                  # not none here because we want to only allow inter eps attn
                self.add_sequence_idx_embed,                ## why attnetion mask has to be non none here? because this is only true if the above is true?
            ]
        ):
            assert attention_mask is not None

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = (
                seq_length_with_past + past_key_values_length
            )

        if position_ids is None and not self.reset_position_index:
            device = (
                input_ids.device
                if input_ids is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length + self.num_sink_tokens,
                dtype=torch.long,
                device=device,
            ) - 1
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length + self.num_sink_tokens)
        elif position_ids is None and self.reset_position_index:
            position_ids = torch.tile(
                torch.arange(
                    seq_length + past_key_values_length + self.num_sink_tokens,
                    device=attention_mask.device
                ),
                (attention_mask.shape[0], 1),
            ) - self.num_sink_tokens
            position_ids[:, :self.num_sink_tokens + 1] = 0
            position_ids = (
                position_ids
                - torch.cummax(position_ids * (1 - attention_mask), dim=-1)[0]
            )[:, past_key_values_length:]
            position_ids = position_ids.long()
        else:
            position_ids = position_ids.view(-1, seq_length + self.num_sink_tokens).long()
        position_ids += self.num_sink_tokens
        position_ids[:, :self.num_sink_tokens] = torch.arange(self.num_sink_tokens)[None]

        sequence_embed = 0.0
        if self.add_sequence_idx_embed:
            sequence_ids = (
                (
                    torch.cumsum(1 - attention_mask[:, past_key_values_length:], dim=-1)
                    + (1 - attention_mask[:, :past_key_values_length]).sum(dim=1, keepdim=True)
                    - (1 - attention_mask[:, self.num_sink_tokens:self.num_sink_tokens+1])
                ).to(attention_mask.device).long()
            )
            sequence_ids[:, :self.num_sink_tokens] = 0
            if self.sequence_embed_type == "linear":
                sequence_embed = self.embed_sequence_pos(sequence_ids)

        if not self.inter_episodes_attention or self.banded_attention or not self.orphan_steps_attention:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length + self.num_sink_tokens),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = None
            if output_attentions:
                raise ValueError(
                    "output_attentions=True is not supported right now for this model"
                )

        position_embed = 0.0
        if self.position_embed_type == "linear":
            position_embed = self.embed_pos(position_ids)

        if self.position_embed_type == "rope":
            position_ids = position_ids
        elif self.add_sequence_idx_embed and self.sequence_embed_type == "rope":
            position_ids = sequence_ids
        else:
            position_ids = None

        # print(position_ids[0])
        # print(attention_mask[0])

        hidden_states = inputs_embeds + sequence_embed + position_embed 

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:                           ## ever true?

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs, past_key_value, output_attentions
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1][..., self.num_sink_tokens:, self.num_sink_tokens:],)

        hidden_states = self.norm(hidden_states[:, self.num_sink_tokens:])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

####################################################################################################

class AutocompLlamaRLModel(LlamaRLModel):
    def __init__(self, config: LlamaRLConfig):
        super().__init__(config)
        self.summary_length = config.summary_length                                                      ## TODO: make configables
        self.segment_length = config.segment_length
        assert self.segment_length > self.summary_length
        if self.summary_length > 0:
            self.embed_summary = nn.Embedding(self.summary_length, config.hidden_size)
        self.detach_prob = config.detach_prob
        self.dont_randomise_segments = config.dont_randomise_segments


    def create_softprompt(self, new_softprompt, softprompt):
        if softprompt is not None:
            if torch.rand(1) < self.detach_prob:                ## this is currently not useful, maybe more efficient if we do the fwd within a nograd() block
                new_softprompt = new_softprompt.detach()
            softprompt = torch.cat([softprompt, new_softprompt], dim=1)
        else:
            if torch.rand(1) < self.detach_prob:
                softprompt = new_softprompt.detach()
            else:
                softprompt = new_softprompt
        return softprompt

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):
        
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(                        ## check this func also
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:                                              ## checkl what comes in here when we are in signle eps setting
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if not self.inter_episodes_attention:
                expanded_attn_mask = create_episodes_mask(
                    attention_mask, dtype, tgt_len=input_shape[-1]
                )

                combined_attention_mask = (
                    expanded_attn_mask
                    if combined_attention_mask is None
                    else expanded_attn_mask + combined_attention_mask
                )

            if not self.orphan_steps_attention:                                                                 ## need to udnerstand what this does
                orphan_mask = (torch.cumsum(1 - attention_mask, dim=1) == 0)[:, None, None].to(dtype)
                orphan_mask.masked_fill_(orphan_mask.to(torch.bool), torch.finfo(dtype).min)
                orphan_mask = orphan_mask[:, :, -input_shape[-1]:].to(device)

                combined_attention_mask = (
                    orphan_mask
                    if combined_attention_mask is None
                    else orphan_mask + combined_attention_mask
                )
        return combined_attention_mask


    def _prepare_position_embeds(
        self,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.FloatTensor,
        past_key_values_length: int,
        seq_length: int,
    ) -> Tuple[Optional[torch.LongTensor], torch.FloatTensor]:
        if position_ids is None and not self.reset_position_index:
            device = inputs_embeds.device

            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ) # - 1                                                                               ## think should remove TODO removed for now
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        elif position_ids is None and self.reset_position_index:
            position_ids = torch.tile(
                torch.arange(
                    seq_length + past_key_values_length,
                    device=attention_mask.device
                ),
                (attention_mask.shape[0], 1),
            )
            # position_ids[:, :self.num_sink_tokens + 1] = 0                    ## check if ok to delete
            position_ids = (
                position_ids
                - torch.cummax(position_ids * (1 - attention_mask), dim=-1)[0]
            )[:, past_key_values_length:]
            position_ids = position_ids.long()
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        position_embed = 0.0
        if self.position_embed_type == "linear":
            position_embed = self.embed_pos(position_ids)

        if self.position_embed_type == "rope":
            position_ids = position_ids
        else:
            position_ids = None

        return position_ids, position_embed


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_ac_outputs=False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        assert position_ids is None
            
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You have to specify inputs_embeds")
        elif input_ids is not None:
            raise ValueError("You have to specify inputs_embeds")
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify inputs_embeds")

        if any(
            [
                not self.inter_episodes_attention,
                self.reset_position_index,
            ]
        ):
            assert attention_mask is not None

        # decoder layers
        all_hidden_states = None # () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        ####################################################################################################

        if self.training or inputs_embeds.size(1) >= self.segment_length:        
            softprompt_decoder_cache = () 
            assert past_key_values is None
            past_key_values_length = 0
            softprompt = None
            traj_embs = []
            # segment_length = self.segment_length
            if use_cache or self.dont_randomise_segments:                          ## if we are using cache we need to use the same segment lengths as in training
                # just use the seg length when doing validation on trajectories
                segment_length = self.segment_length
            else:
                # sample a random segment length within 20% of the self.segment_length if not using cache because that means we are training
                segment_length = int(self.segment_length * (0.8 + 0.4 * torch.rand(1).item()))
            seq_segments = list(torch.split(inputs_embeds, segment_length, dim=1))

            # just pre-decide the 0 or non-zero summary lens for all segments so we can use generic logic to process each segment 
            per_segment_summary_lens = [0 if segment.size(1) < segment_length else self.summary_length for segment in seq_segments]

            # if we only got one segment and it is exactly equal to the segment length meaning  we will make a summary
            # then we need to make a kv cache for this segment's summary after processing the segment and outputting the summary 
            # we can do this by just adding a dummy empty segment to the list of segments so that the code below can be used to process the summary
            if seq_segments[-1].size(1) == segment_length:
                seq_segments.append(torch.zeros_like(seq_segments[0])[:, :0])
                per_segment_summary_lens.append(0)

            for i, segment in enumerate(seq_segments):
                segment_summary_len = per_segment_summary_lens[i]
                summary_token_ids = torch.arange(segment_summary_len, dtype=torch.long, device=segment.device).unsqueeze(0).expand(segment.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(segment.dtype)

                _, segment_seq_length, _ = segment.shape 
                
                # decide how much to remove from the back and front of the embeddings generated later to get the summaries vs the segment embeds
                remove_from_front_idxs = 0

                segment = torch.cat([segment, summary_token_embeds], dim=1)
                segment_seq_length += segment_summary_len          
                remove_from_back_idxs = segment_summary_len

                if softprompt is not None:
                    segment = torch.cat([softprompt, segment], dim=1)
                    segment_seq_length += softprompt.size(1)
                    remove_from_front_idxs = softprompt.size(1)
            
                position_ids, position_embed = self._prepare_position_embeds(
                    None, attention_mask, segment, 0, segment_seq_length
                )
                if not self.inter_episodes_attention or not self.orphan_steps_attention:            ## only gets exec if we are in the single eps setting - use this for now
                    attention_mask = self._prepare_decoder_attention_mask(
                        attention_mask,
                        (batch_size, segment_seq_length),
                        segment,
                        0,
                    )
                else:
                    attention_mask = None
                
                traj_emb_i = segment + position_embed 
                
                for idx, decoder_layer in enumerate(self.layers):

                    # assert attention_mask is None

                    layer_outputs = decoder_layer(
                        traj_emb_i,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                    traj_emb_i = layer_outputs[0]
                    if use_cache:
                        kv_cache = layer_outputs[2 if output_attentions else 1]
                        # remove the softprompt generayion kv from kv_cache, and separate the softprompt encoding kv from the rest
                        num_steps_cache = kv_cache[0].shape[2]
                        next_decoder_cache += (
                            (kv_cache[0][:, :, remove_from_front_idxs:num_steps_cache-remove_from_back_idxs], kv_cache[1][:, :, remove_from_front_idxs:num_steps_cache-remove_from_back_idxs]),
                        )
                        # if final segment being processed, also extract the softprompt kv cache computed at the start of this segment
                        if i == len(seq_segments) - 1:
                            softprompt_decoder_cache += (
                                (kv_cache[0][:, :, :remove_from_front_idxs], kv_cache[1][:, :, :remove_from_front_idxs]),
                            )

                traj_emb_i = self.norm(traj_emb_i)              ## WOULD REMOVE THIS

                if segment_summary_len > 0:
                    new_softprompt = traj_emb_i[:, -self.summary_length:]
                    traj_emb_i = traj_emb_i[:, remove_from_front_idxs:-self.summary_length]
                    # traj_emb_i = self.norm(traj_emb_i)

                    # if softprompt is not None:
                    #     if torch.rand(1) < self.detach_prob:                ## this is currently not useful, maybe more efficient if we do the fwd within a nograd() block
                    #         new_softprompt = new_softprompt.detach()
                    #     softprompt = torch.cat([softprompt, new_softprompt], dim=1)
                    # else:
                    #     if torch.rand(1) < self.detach_prob:
                    #         softprompt = new_softprompt.detach()
                    #     else:
                    #         softprompt = new_softprompt
                    softprompt = self.create_softprompt(new_softprompt, softprompt)
                else:
                    traj_emb_i = traj_emb_i[:, remove_from_front_idxs:]
                    # traj_emb_i = self.norm(traj_emb_i)

                traj_embs.append(traj_emb_i)

            hidden_states = torch.cat(traj_embs, dim=1)

        else:
            past_key_values_length = 0
            make_up_fifo_length = 0
            if past_key_values is not None:
                past_key_values_length = past_key_values["past_key_values"].shape[4]                
                past_softprompt_key_values_length = 0
                if past_key_values["softprompt"] is not None:
                    past_softprompt_key_values_length = past_key_values["softprompt"].shape[4]
                    # if past_key_values_length // self.segment_length > past_softprompt_key_values_length/self.summary_length:
                        # make_up_fifo_length = (past_key_values_length // self.segment_length - past_softprompt_key_values_length//self.summary_length) * self.summary_length

            softprompt = None
            softprompt_extracted = None
            make_summary = False
            # if another segment is about to end, let's make a summary
            if (past_key_values_length + seq_length) % self.segment_length==0:
                make_summary = True
                summary_token_ids = torch.arange(self.summary_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(inputs_embeds.dtype)

                inputs_embeds = torch.cat([inputs_embeds, summary_token_embeds], dim=1)
                seq_length += self.summary_length 

            # the index where we want to pick up the kv cache from after excluding the summarised parts
            starting_idx_of_last_segment = (past_key_values_length // self.segment_length) * self.segment_length
            past_key_value = None

            if past_key_values is not None:
                if past_key_values_length > starting_idx_of_last_segment:
                    recent_segment_kv_cache = past_key_values["past_key_values"][:, :, :, :, starting_idx_of_last_segment:]
                    past_key_value = torch.cat([past_key_values["softprompt"], recent_segment_kv_cache], dim=4) if past_key_values["softprompt"] is not None else recent_segment_kv_cache
                elif past_key_values_length == starting_idx_of_last_segment:                     ## when there's no unsummarised kv cache to use
                    past_key_value = past_key_values["softprompt"] if past_key_values["softprompt"] is not None else None
                else:
                    raise ValueError("past_key_values_length is less than starting_idx_of_last_segment")

            past_key_values_length_now = past_key_value.shape[4] if past_key_value is not None else 0

            position_ids, position_embed = self._prepare_position_embeds(
                position_ids, attention_mask, inputs_embeds, past_key_values_length_now + make_up_fifo_length, seq_length
            )

            # # TODO: we should make attn mask more generic
            if not self.inter_episodes_attention or not self.orphan_steps_attention:
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                attention_mask = None

            hidden_states = inputs_embeds + position_embed 

            if output_attentions:
                raise ValueError(
                    "output_attentions=True is not supported right now for this model"
                )
                # manually create a causal mask, where things being attended to have value 1 in the mask, and others have value -inf in torch
                inputs_embeds_shape_now  = hidden_states.shape
                atten_mask_shape = (inputs_embeds_shape_now[0], 1, inputs_embeds_shape_now[1], inputs_embeds_shape_now[1] + past_key_values_length_now)                
                attention_mask = torch.ones(atten_mask_shape, device=inputs_embeds.device)
                # make it causal
                for i in range(inputs_embeds_shape_now[1]):
                    attention_mask[:, :, i, i+past_key_values_length_now+1:] = -float('inf')

            for idx, decoder_layer in enumerate(self.layers):

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value[idx] if past_key_value is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (
                        layer_outputs[2 if output_attentions else 1],
                    )
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)               ## WOULD REMOVE THIS
            
            softprompt_decoder_cache = None
            if make_summary:    ## if we made summary tokens, we also want to make their kv cache so that we can just use these for future inference
                softprompt_decoder_cache = ()
                # separate the softprompt from the hidden states
                softprompt_extracted = hidden_states[:, -self.summary_length:]
                hidden_states = hidden_states[:, :-self.summary_length]
                # hidden_states = self.norm(hidden_states)

                older_prompt_kv_cache = past_key_values["softprompt"] if past_key_values["softprompt"] is not None else None
                
                # add position ids for the softprompt
                sp_position_ids, position_embed = self._prepare_position_embeds(
                    None,
                    None,
                    softprompt_extracted,
                    (older_prompt_kv_cache.shape[4] + make_up_fifo_length) if older_prompt_kv_cache is not None else 0,                                             
                    self.summary_length
                )
                softprompt = softprompt_extracted + position_embed

                # now we just run these softprompt vecs through the model with only any previous summaries as the cache
                # to be able to generate the kv cache for the new softprompt and add it to the softprompt kv cache from previous memories
                for idx, decoder_layer in enumerate(self.layers):


                    layer_outputs = decoder_layer(
                        softprompt,
                        attention_mask=None, 
                        position_ids=sp_position_ids,
                        past_key_value=older_prompt_kv_cache[idx] if older_prompt_kv_cache is not None else None,
                        output_attentions=False,             ## TODO: so this will become wrong if we want to viz attention, avoid this by setting false -.-            
                        use_cache=use_cache,
                    )

                    softprompt = layer_outputs[0]               ## need to comment this to eval old models

                    softprompt_decoder_cache += (
                        layer_outputs[1],                   # using [1] because output_attn=false above
                    )
            # else:
            #     hidden_states = self.norm(hidden_states)

                

            if return_ac_outputs:
                next_cache = next_decoder_cache if use_cache else None
                return ACOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values={"past_key_values": next_cache, "softprompt": softprompt_decoder_cache},
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                    softprompt=softprompt_extracted,
                )
        ################# FINAL RETURN VALUES COMPUTAITON ########################################################

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values={"past_key_values": next_cache, "softprompt": softprompt_decoder_cache},
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def act(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_ac_outputs=False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        assert position_ids is None
            
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You have to specify inputs_embeds")
        elif input_ids is not None:
            raise ValueError("You have to specify inputs_embeds")
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify inputs_embeds")

        if any(
            [
                not self.inter_episodes_attention,
                self.reset_position_index,
            ]
        ):
            assert attention_mask is not None

        # decoder layers
        all_hidden_states = None # () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values["past_key_values"].shape[4] 
            past_softprompt_key_values_length = 0
            if past_key_values["softprompt"] is not None:
                past_softprompt_key_values_length = past_key_values["softprompt"].shape[4]

        softprompt = None
        softprompt_extracted = None
        make_summary = False
        # if another segment is about to end, let's make a summary
        if (past_key_values_length + seq_length) % self.segment_length==0:
            make_summary = True
            summary_token_ids = torch.arange(self.summary_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.size(0), -1)
            summary_token_embeds = self.embed_summary(summary_token_ids).to(inputs_embeds.dtype)

            inputs_embeds = torch.cat([inputs_embeds, summary_token_embeds], dim=1)
            seq_length += self.summary_length 

        attention_mask = None

        # the index where we want to pick up the kv cache from after excluding the summarised parts
        # starting_idx_of_last_segment = (past_key_values_length // self.segment_length) * self.segment_length
        past_key_value = None

        assert past_key_values_length<256   # check this
        
        if past_key_values is not None:
            if past_key_values_length > 0:
                recent_segment_kv_cache = past_key_values["past_key_values"] # [:, :, :, :, starting_idx_of_last_segment:]
                past_key_value = torch.cat([past_key_values["softprompt"], recent_segment_kv_cache], dim=4) if past_key_values["softprompt"] is not None else recent_segment_kv_cache
            elif past_key_values_length == 0:                     ## when there's no unsummarised kv cache to use
                past_key_value = past_key_values["softprompt"] if past_key_values["softprompt"] is not None else None
            else:
                raise ValueError("past_key_values_length is less than starting_idx_of_last_segment")

        past_key_values_length_now = past_key_value.shape[4] if past_key_value is not None else 0

        position_ids, position_embed = self._prepare_position_embeds(
            position_ids, attention_mask, inputs_embeds, past_key_values_length_now, seq_length
        )
        
        hidden_states = inputs_embeds + position_embed 

        if output_attentions:
            # manually create a causal mask, where things being attended to have value 1 in the mask, and others have value -inf in torch
            raise ValueError("output_attentions not supported in act")
            inputs_embeds_shape_now  = hidden_states.shape
            atten_mask_shape = (inputs_embeds_shape_now[0], 1, inputs_embeds_shape_now[1], inputs_embeds_shape_now[1] + past_key_values_length_now)                
            attention_mask = torch.ones(atten_mask_shape, device=inputs_embeds.device)
            # make it causal
            for i in range(inputs_embeds_shape_now[1]):
                attention_mask[:, :, i, i+past_key_values_length_now+1:] = -float('inf')

        for idx, decoder_layer in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value[idx] if past_key_value is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)                      ## WOULD REMOVE THIS
        
        softprompt_decoder_cache = None
        if make_summary:    ## if we made summary tokens, we also want to make their kv cache so that we can just use these for future inference
            softprompt_decoder_cache = ()
            # separate the softprompt from the hidden states
            softprompt_extracted = hidden_states[:, -self.summary_length:]
            hidden_states = hidden_states[:, :-self.summary_length]

            older_prompt_kv_cache = past_key_values["softprompt"] if past_key_values["softprompt"] is not None else None
            
            # add position ids for the softprompt
            sp_position_ids, position_embed = self._prepare_position_embeds(
                None,
                None,
                softprompt_extracted,
                older_prompt_kv_cache.shape[4] if older_prompt_kv_cache is not None else 0,                                             
                self.summary_length
            )
            softprompt = softprompt_extracted + position_embed

            # now we just run these softprompt vecs through the model with only any previous summaries as the cache
            # to be able to generate the kv cache for the new softprompt and add it to the softprompt kv cache from previous memories
            for idx, decoder_layer in enumerate(self.layers):


                layer_outputs = decoder_layer(
                    softprompt,
                    attention_mask=None, 
                    position_ids=sp_position_ids,
                    past_key_value=older_prompt_kv_cache[idx] if older_prompt_kv_cache is not None else None,
                    output_attentions=False,             ## TODO: so this will become wrong if we want to viz attention, avoid this by setting false -.-            
                    use_cache=use_cache,
                )

                softprompt = layer_outputs[0]               ## need to comment this to eval old models

                softprompt_decoder_cache += (
                    layer_outputs[1],                   # using [1] because output_attn=false above
                )
                
        next_cache = next_decoder_cache if use_cache else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values={"past_key_values": next_cache, "softprompt": softprompt_decoder_cache},
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

####################################################################################################

class RMTLlamaRLModel(AutocompLlamaRLModel):
    def create_softprompt(self, new_softprompt, softprompt):
        return new_softprompt

####################################################################################################
