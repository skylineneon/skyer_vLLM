from vllm import ModelRegistry
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import modeling_auto
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import is_pp_missing_parameter, make_layers


class SkyerConfig(PretrainedConfig):
    model_type = "skyer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = kwargs.get("num_hidden_layers")
        self.hidden_size = kwargs.get("hidden_size")
        self.intermediate_size = kwargs.get("intermediate_size")
        self.num_attention_heads = kwargs.get("num_attention_heads")
        self.num_key_value_heads = kwargs.get("num_key_value_heads")
        self.max_pos_len = kwargs.get("max_pos_len")
        self.vocab_size = kwargs.get("vocab_size")
        self.cache_max_batch_size = kwargs.get("cache_max_batch_size")
        self.cache_max_seq_len = kwargs.get("cache_max_seq_len")
        self.use_cache = kwargs.get("use_cache")
        self.head_dim = kwargs.get("head_dim")

        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3


class SkyerAttention(nn.Module):

    def __init__(self,
                 config: SkyerConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None
                 ):

        super().__init__()

        self._config = config
        self._cache_config = cache_config

        self._head_size = self._config.intermediate_size // self._config.num_attention_heads
        self._q_size = self._config.num_attention_heads * self._head_size
        self._kv_size = self._config.num_key_value_heads * self._head_size
        self._scaling = self._head_size**-0.5

        self._qkv_proj = QKVParallelLinear(
            hidden_size=self._config.hidden_size,
            head_size=self._head_size,
            total_num_heads=self._config.num_attention_heads,
            total_num_kv_heads=self._config.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
        )

        self._ow = ColumnParallelLinear(
            input_size=self._config.intermediate_size,
            output_size=self._config.hidden_size,
            bias=True,
            quant_config=quant_config,
        )

        self._attn = Attention(
            num_heads=self._config.num_attention_heads,
            head_size=self._head_size,
            scale=self._scaling,
            num_kv_heads=self._config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config
        )

        self._rotary_emb = get_rope(
            self._head_size,
            rotary_dim=self._head_size,
            max_position=self._config.max_pos_len,
            base=50000
        )

    def forward(self,
                position_ids: torch.Tensor,
                hidden_states: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata: AttentionMetadata):
        _qkv, _ = self._qkv_proj(hidden_states)
        _q, _k, _v = _qkv.split(
            [self._q_size, self._kv_size, self._kv_size], dim=-1)
        _q, _k = self._rotary_emb(position_ids, _q, _k)
        _attn_output = self._attn(_q, _k, _v, kv_cache, attn_metadata)
        _output, _ = self._ow(_attn_output)
        return _output


class SkyerFFN(nn.Module):

    def __init__(self,
                 config: SkyerConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        self._config = config
        self._quant_config = quant_config

        self._gate_up_proj = MergedColumnParallelLinear(
            input_size=self._config.hidden_size,
            output_sizes=[self._config.intermediate_size] * 2,
            bias=True,
            quant_config=quant_config
        )

        self._down_proj = ColumnParallelLinear(
            input_size=self._config.intermediate_size,
            output_size=self._config.hidden_size,
            bias=True,
            quant_config=self._quant_config
        )

        self._act_fn = SiluAndMul()

    def forward(self, hidden_states):
        _hidden_states = hidden_states

        _gate_up, _ = self._gate_up_proj(_hidden_states)
        _hidden_states = self._act_fn(_gate_up)
        _hidden_states, _ = self._down_proj(_hidden_states)

        return _hidden_states


class SkyerLayer(nn.Module):

    def __init__(self,
                 config: SkyerConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        self._config = config
        self._cache_config = cache_config
        self._quant_config = quant_config

        self._input_norm = RMSNorm(
            self._config.hidden_size,
            eps=1e-5
        )

        self._att_layer = SkyerAttention(
            config=self._config,
            cache_config=self._cache_config,
            quant_config=self._quant_config
        )

        self._att_norm = RMSNorm(
            self._config.hidden_size,
            eps=1e-5)

        self._ffn_layer = SkyerFFN(
            config=self._config,
            quant_config=self._quant_config
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:

        _hidden_states = hidden_states
        _position_ids = position_ids
        _kv_cache = kv_cache
        _attn_metadata = attn_metadata

        _residual = residual
        if _residual is None:
            _residual = _hidden_states
            _hidden_states = self._input_norm(
                _hidden_states
            )
        else:
            _hidden_states, _residual = self._input_norm(
                _hidden_states,
                _residual
            )

        _hidden_states = self._att_layer(
            position_ids=_position_ids,
            hidden_states=_hidden_states,
            kv_cache=_kv_cache,
            attn_metadata=_attn_metadata,
        )

        _hidden_states, _residual = self._att_norm(
            _hidden_states,
            _residual
        )

        _hidden_states = self._ffn_layer(
            _hidden_states
        )

        return _hidden_states, _residual


class SkyerModel(nn.Module):

    def __init__(self,
                 config: SkyerConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None,
                 ):
        super().__init__()

        self._config = config
        self._cache_config = cache_config
        self._quant_config = quant_config

        self._emb = VocabParallelEmbedding(
            self._config.vocab_size,
            self._config.hidden_size
        )

        self._start_layer, self._end_layer, self._tf_layers = make_layers(
            self._config.num_hidden_layers,
            lambda prefix: SkyerLayer(
                config=self._config,
                cache_config=self._cache_config,
                quant_config=self._quant_config
            ),
            prefix=""
        )

        self._feat_norm = RMSNorm(
            self._config.hidden_size,
            eps=1e-5
        )

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata):

        _input_ids = input_ids
        _position_ids = positions
        _kv_caches = kv_caches
        _attn_metadata = attn_metadata

        if get_pp_group().is_first_rank:
            _hidden_states = self._emb(_input_ids)
            _residual = None

        for _i in range(self._start_layer, self._end_layer):
            _layer = self._tf_layers[_i]
            _hidden_states, _residual = _layer(
                position_ids=_position_ids,
                hidden_states=_hidden_states,
                kv_cache=_kv_caches[_i - self._start_layer],
                attn_metadata=_attn_metadata,
                residual=_residual
            )

        _hidden_states, _ = self._feat_norm(
            _hidden_states,
            _residual
        )
        # _hidden_states = _hidden_states+_residual
        return _hidden_states


class SkyerForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: SkyerConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:

        super().__init__()

        self._config = config
        self._cache_config = cache_config
        self._quant_config = quant_config
        self._lora_config = lora_config

        self._model = SkyerModel(
            config=self._config,
            cache_config=self._cache_config,
            quant_config=self._quant_config)

        self._lm_head = self._model._emb

        self._logits_processor = LogitsProcessor(self._config.vocab_size)
        self._sampler = Sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                ) -> torch.Tensor:

        _hidden_states = self._model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata)

        return _hidden_states

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata,
                       ) -> Optional[torch.Tensor]:
        _logits = self._logits_processor(
            self._lm_head,
            hidden_states,
            sampling_metadata)

        return _logits

    def sample(self,
               logits: torch.Tensor,
               sampling_metadata: SamplingMetadata,
               ) -> Optional[SamplerOutput]:
        _next_tokens = self._sampler(
            logits,
            sampling_metadata
        )
        return _next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        _params_dict = dict(self.named_parameters())
        for _k,v in self.named_parameters():
            print(_k)
        print("=============================")
        for _name, _loaded_weight in weights:
            print(_name)

            # _name = _name.replace("_tf_layer._layers", "_model._tf_layers")

            # if _name.find("_qw") > 0:
            #     _name = _name.replace("_qw", "_qkv_proj")
            #     _param = _params_dict[_name]
            #     _weight_loader = _param.weight_loader
            #     _weight_loader(_param, _loaded_weight, "q")

            # elif _name.find("_kw") > 0:
            #     print("........")
            #     _name = _name.replace("_kw", "_qkv_proj")
            #     _param = _params_dict[_name]
            #     _weight_loader = _param.weight_loader
            #     _weight_loader(_param, _loaded_weight, "k")

            # elif _name.find("_vw") > 0:
            #     _name = _name.replace("_vw", "_qkv_proj")
            #     _param = _params_dict[_name]
            #     _weight_loader = _param.weight_loader
            #     _weight_loader(_param, _loaded_weight, "v")

            # elif _name.find("_w0") > 0:
            #     _name = _name.replace("_w0", "_gate_up_proj")
            #     _param = _params_dict[_name]
            #     _weight_loader = _param.weight_loader
            #     _weight_loader(_param, _loaded_weight, 1)

            # elif _name.find("_w1") > 0:
            #     _name = _name.replace("_w1", "_gate_up_proj")
            #     _param = _params_dict[_name]
            #     _weight_loader = _param.weight_loader
            #     _weight_loader(_param, _loaded_weight, 0)
            # else:
            #     _name = _name.replace("_emb", "_model._emb")
            #     _name = _name.replace("_w2", "_down_proj")
            #     _name = _name.replace("_att_norm._w", "_input_norm.weight")
            #     _name = _name.replace("_ffn_norm._w", "_att_norm.weight")
            #     _name = _name.replace(
            #         "_tf_layer._out_norm._w", "_model._feat_norm.weight")

            #     _param = _params_dict[_name]
            #     _weight_loader = getattr(
            #         _param, "weight_loader", default_weight_loader)
            #     _weight_loader(_param, _loaded_weight)


AutoConfig.register("skyer", SkyerConfig)
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["skyer"] = "SkyerForCausalLM"

ModelRegistry.register_model("SkyerForCausalLM", SkyerForCausalLM)
