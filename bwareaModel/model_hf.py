from typing import Any, Optional, Tuple, Union, List

import math
import torch
# import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizerFast, LlamaConfig, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils.import_utils import is_torch_fx_available
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_12

from dataclasses import dataclass
from transformers.utils import ModelOutput

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_12:
        import torch.fx
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


@dataclass
class IntentionModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    action_index: Optional[Tuple[torch.LongTensor]] = None

@dataclass
class CausalIntentionLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    action_index: Optional[Tuple[torch.LongTensor]] = None


class IntentionForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        model = IntentionVQModel
        self.model = model(config)

    def reset_action_info(self):
        for key, _ in self.model.action_info.items():
            self.model.action_info[key].clear()
    
    def get_action_info(self):
        return self.model.action_info
    
    def set_action_sampling(self, greedy=True, temp=1.0):
        self.model.deterministic = greedy
        self.model.action_code_book.set_temp(temp)

    def forward_inverse(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalIntentionLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward_vqvae(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return CausalIntentionLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            action_index=outputs.action_index,
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        action_idx = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalIntentionLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            action_idx=action_idx,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return CausalIntentionLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            action_index=outputs.action_index,
        )
    
    def forward_policy(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalIntentionLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward_policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        logits = logits.float()

        loss = None
        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return CausalIntentionLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            action_index=outputs.action_index,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values['transformer_kv'][0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
       
class VQ(nn.Module):
    def __init__(self, num_code, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5, temperature=1.0):
        super(VQ, self).__init__()

        self.dim = embedding_dim
        self.n_embed = num_code
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        
        self.embed = nn.Embedding(self.n_embed, self.dim)
    
    def set_temp(self, temp=1.0):
        self.temperature = temp
    
    def set_training(self, is_train=True):
        self.training = is_train
        
    def reset(self):
        self.embed.weight.data.uniform_(-1., 1.)
        
    def get_embedding(self):
        mean = self.embed.weight.data.mean()
        std_ = self.embed.weight.data.std()
        max_ = self.embed.weight.data.max()
        min_ = self.embed.weight.data.min()
        return {'emb/mean': mean, 'emb/std': std_, 'emb/max': max_, 'emb/min': min_}

    def forward(self, inputs, masks=None, use_cache=False):
        flatten = inputs.reshape(-1, self.dim)
        
        dist = (torch.sum(flatten ** 2, dim=1, keepdim=True)
                + torch.sum(self.embed.weight ** 2, dim=1)
                - 2 * torch.matmul(flatten, self.embed.weight.t()))

        _, embed_ind = (-dist).max(1)
        embed_onehot = torch.zeros(embed_ind.shape[0], self.n_embed, device=inputs.device, dtype=self.embed.weight.dtype)
        embed_onehot.scatter_(1, embed_ind.unsqueeze(1), 1)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = torch.matmul(embed_onehot, self.embed.weight).view(*inputs.shape)
            
        # avg_probs = torch.mean(embed_onehot, dim=0)
        # preplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantize = inputs + (quantize - inputs).detach()
        return quantize, embed_ind

    
    def forward_with_idx(self, inputs, deterministic=True, epsilon_greedy=False, action_idx=None):
        flatten = inputs.reshape(-1, self.n_embed)
        if deterministic:
            _, embed_ind = (flatten).max(1)
            # if epsilon_greedy and torch.rand(1) > 0.95:
                # embed_ind = Categorical(F.softmax(flatten, dim=-1, dtype=torch.float)).sample()
        else:
            flatten = flatten / self.temperature
            embed_ind = Categorical(F.softmax(flatten, dim=-1, dtype=torch.float)).sample()
            # embed_ind = Categorical(F.softmax(flatten, dim=-1)).sample()
        # embed_onehot = torch.zeros(embed_ind.shape[0], self.n_embed, device=inputs.device)
        embed_onehot = torch.zeros(embed_ind.shape[0], self.n_embed, device=inputs.device, dtype=self.embed.weight.dtype)
        if action_idx is not None and not isinstance(action_idx, torch.LongTensor) and action_idx > -1:
            embed_onehot[:, action_idx] = 1
        elif isinstance(action_idx, torch.LongTensor):
            action_idx = action_idx.reshape(-1)
            embed_onehot.scatter_(1, action_idx.unsqueeze(1), 1)
        else:
            embed_onehot.scatter_(1, embed_ind.unsqueeze(1), 1)
        # quantize = self.embed(embed_ind.detach())
        quantize = torch.matmul(embed_onehot, self.embed.weight).reshape(inputs.shape[0], inputs.shape[1], self.dim)
        return quantize, embed_ind.reshape(inputs.shape[0], inputs.shape[1])
    

class IntentionVQModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)
        self.action_info = dict(
            action_idx=[],
        )
        self.deterministic = False
        self.policy_temperature = 1.0
        self.epsilon_greedy = False

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.enc_layer_num = config.num_hidden_layers // 2
        self.dec_layer_num = config.num_hidden_layers - self.enc_layer_num
        self.action_dim = config.action_dim
        self.lm_head_bias = False

        self.num_code = config.num_code # Need to add num code in config
        self.num_dim = config.num_dim # add num_dim into config
        self.policy_layer_num = config.num_hidden_layers

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # self.dec_layer_num = 1
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx),
                h=nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]),
                ln_f=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        )
        self.transformer_action_enc = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx),
                h=nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(self.enc_layer_num)]),
                ln_f=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        )
        # Policy
        self.transformer_policy = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx),
                # h=nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(self.policy_layer_num)]),
                h=nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(self.policy_layer_num)]),
                ln_f=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        )
        self.transformer_policy_layer = nn.Linear(config.hidden_size, self.num_code, bias=self.lm_head_bias)
        
        self.press_layer = nn.Linear(config.hidden_size, self.num_dim, bias=self.lm_head_bias)        
        self.action_code_book = VQ(self.num_code, self.num_dim, self.policy_temperature)
        self.unpress_layer = nn.Linear(self.num_dim, config.hidden_size, bias=self.lm_head_bias)    
        self.dynamics_layer = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.dynamics_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.mask_cache: Optional[torch.Tensor] = None
        self.action_range = [-10., 10.]
        self.eps = 1e-4
        
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def set_action_info(self, action_idx):
        self.action_info["action_idx"].append(action_idx)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        action_idx=None,
    ) -> Union[Tuple, IntentionModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "[Warning!]: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        past_key_values_policy = None
        past_key_values_inverse_dynamics = None
        past_key_values_transformer = None

        if use_cache:
            if past_key_values is None:
                past_key_values = dict(
                    policy_kv=None,
                    inverse_dynamics_kv=None,
                    transformer_kv=None,
                )
            use_legacy_cache = not isinstance(past_key_values["policy_kv"], Cache)
            if use_legacy_cache:
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values["policy_kv"])
                past_key_values_inverse_dynamics = DynamicCache.from_legacy_cache(past_key_values["inverse_dynamics_kv"])
                past_key_values_transformer = DynamicCache.from_legacy_cache(past_key_values["transformer_kv"])
            else:
                past_key_values_policy = past_key_values["policy_kv"]
                past_key_values_inverse_dynamics = past_key_values["inverse_dynamics_kv"]
                past_key_values_transformer = past_key_values["transformer_kv"]
            past_key_values_length = past_key_values_policy.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache_policy = None
        next_decoder_cache_inverse_dynamic = None
        next_decoder_cache_transformer = None
        
        # inverse dynamics
        x_a_inverse = self.transformer_action_enc.wte(input_ids) 
        attention_mask_action_inverse = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x_a_inverse, past_key_values_length
        )
        for block_a_inverse in self.transformer_action_enc.h:
            layer_outputs = block_a_inverse(
                x_a_inverse,
                attention_mask=attention_mask_action_inverse,
                position_ids=position_ids,
                past_key_value=past_key_values_inverse_dynamics,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # print(layer_outputs, layer_outputs[0].shape)
            x_a_inverse = layer_outputs[0]
            if use_cache:
                next_decoder_cache_inverse_dynamic = layer_outputs[2 if output_attentions else 1]

        x_a_inverse = self.transformer_action_enc.ln_f(x_a_inverse)
        x_a_inverse[:, :-1] = x_a_inverse[:, 1:].clone()
        x_a_inverse = self.press_layer(x_a_inverse)
        x_a_inverse, _ = self.action_code_book(x_a_inverse, masks=attention_mask, use_cache=use_cache)
        x_a_inverse = self.unpress_layer(x_a_inverse)

        # policy bc
        x_a_policy = self.transformer_policy.wte(input_ids) 
        attention_mask_xa = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x_a_policy, past_key_values_length
        )
        for block_a_policy in self.transformer_policy.h:
            layer_outputs = block_a_policy(
                x_a_policy,
                attention_mask=attention_mask_xa,
                position_ids=position_ids,
                past_key_value=past_key_values_policy,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x_a_policy = layer_outputs[0]
            if use_cache:
                next_decoder_cache_policy = layer_outputs[2 if output_attentions else 1]
        
        x_a_policy = self.transformer_policy.ln_f(x_a_policy)
        x_a_policy = self.transformer_policy_layer(x_a_policy)

        x_a_policy, action_idx = self.action_code_book.forward_with_idx(
            x_a_policy, 
            deterministic=self.deterministic, 
            epsilon_greedy=self.epsilon_greedy, 
            action_idx=action_idx
        )
        x_a_policy = self.unpress_layer(x_a_policy)

        self.set_action_info(action_idx=action_idx[:, -1:])

        x = self.transformer.wte(input_ids)
        mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x, past_key_values_length
        )
        for num_block, block in enumerate(self.transformer.h):
            if num_block == self.enc_layer_num:
                x_a_policy = self.dynamics_norm(self.dynamics_layer(torch.cat([x, x_a_policy], dim=-1)))
                x_policy_s = x_a_policy + x

                # x = x_policy_s
                # use_cache to accelerate
                if use_cache:
                    x= x_policy_s
                else:
                    print("========Use ID ===================")
                    x_a_inverse = self.dynamics_norm(self.dynamics_layer(torch.cat([x, x_a_inverse], dim=-1)))
                    x_a_inverse_s = x_a_inverse + x
                    
                    x = torch.cat([x_a_inverse_s, x_policy_s], dim=1)  # bs, 2*lens, dim
                    # cos = torch.cat([cos, cos], dim=0)  # 2*lens, dim
                    # sin = torch.cat([sin, sin], dim=0)  # 2*lens, dim
                    if position_ids is not None:
                        position_ids = torch.cat([position_ids, position_ids], dim=-1)  # 2*lens
                        
                    mask_sp = torch.cat([
                            torch.ones((seq_length, seq_length), dtype=torch.bool).tril(), 
                            torch.zeros((seq_length, seq_length), dtype=torch.bool)
                        ],dim=-1
                    )
                    mask_sp = mask_sp.unsqueeze(0).unsqueeze(0).to(x.device)

                    mask_sa = torch.cat([
                        (torch.ones((seq_length, seq_length)).tril() - torch.eye(seq_length)).bool(), 
                        torch.eye(seq_length, dtype=torch.bool)
                        ], dim=-1
                    )
                    mask_sa = mask_sa.unsqueeze(0).unsqueeze(0).to(x.device)
                    mask = torch.cat([mask_sp, mask_sa], dim=2)
                    
                    if attention_mask is not None:
                        mask = mask.repeat(attention_mask.shape[0], 1, 1, 1)
                        pad_mask_ = torch.cat([attention_mask, attention_mask], dim=1)
                        mask = mask.permute(0, 1, 3, 2)
                        mask[pad_mask_.unsqueeze(1) == 0] = False
                        mask[:, :, torch.arange(seq_length * 2), torch.arange(seq_length * 2)] = True
                        mask = mask.permute(0, 1, 3, 2)
                    
            layer_outputs = block(
                x,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_values_transformer,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x = layer_outputs[0]
            if use_cache:
                next_decoder_cache_transformer = layer_outputs[2 if output_attentions else 1]
        
        if not use_cache:
            x = x[:, seq_length:]
        x = self.transformer.ln_f(x)

        if use_cache:
            next_cache_policy = next_decoder_cache_policy.to_legacy_cache() if use_legacy_cache else next_decoder_cache_policy
            next_cache_inverse_dynamic = next_decoder_cache_inverse_dynamic.to_legacy_cache() if use_legacy_cache else next_decoder_cache_inverse_dynamic
            next_cache_transformer = next_decoder_cache_transformer.to_legacy_cache() if use_legacy_cache else next_decoder_cache_transformer

            next_cache = dict(
                policy_kv=next_cache_policy,
                inverse_dynamics_kv=next_cache_inverse_dynamic,
                transformer_kv=next_cache_transformer,
            )
        else:
            next_cache = None

        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns, action_idx] if v is not None)
        return IntentionModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            action_index=action_idx,
        )
    
    def forward_policy(
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
        action_idx=None,
    ) -> Union[Tuple, IntentionModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "[Warning!]: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        past_key_values_policy = None
        # past_key_values_inverse_dynamics = None
        past_key_values_transformer = None

        if use_cache:
            if past_key_values is None:
                past_key_values = dict(
                    policy_kv=None,
                    # inverse_dynamics_kv=None,
                    # transformer_kv=None,
                )
            use_legacy_cache = not isinstance(past_key_values["policy_kv"], Cache)
            if use_legacy_cache:
                past_key_values_policy = DynamicCache.from_legacy_cache(past_key_values["policy_kv"])
                # past_key_values_inverse_dynamics = DynamicCache.from_legacy_cache(past_key_values["inverse_dynamics_kv"])
                # past_key_values_transformer = DynamicCache.from_legacy_cache(past_key_values["transformer_kv"])
            else:
                past_key_values_policy = past_key_values["policy_kv"]
                # past_key_values_inverse_dynamics = past_key_values["inverse_dynamics_kv"]
                # past_key_values_transformer = past_key_values["transformer_kv"]
            past_key_values_length = past_key_values_policy.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache_policy = None
        # next_decoder_cache_inverse_dynamic = None
        # next_decoder_cache_transformer = None
        
        # inverse dynamics
        # x_a_inverse = self.transformer_action_enc.wte(input_ids) 
        # attention_mask_action_inverse = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), x_a_inverse, past_key_values_length
        # )
        # for block_a_inverse in self.transformer_action_enc.h:
        #     layer_outputs = block_a_inverse(
        #         x_a_inverse,
        #         attention_mask=attention_mask_action_inverse,
        #         position_ids=position_ids,
        #         past_key_value=past_key_values_inverse_dynamics,
        #         output_attentions=output_attentions,
        #         use_cache=use_cache,
        #     )
        #     # print(layer_outputs, layer_outputs[0].shape)
        #     x_a_inverse = layer_outputs[0]
        #     if use_cache:
        #         next_decoder_cache_inverse_dynamic = layer_outputs[2 if output_attentions else 1]

        # x_a_inverse = self.transformer_action_enc.ln_f(x_a_inverse)
        # x_a_inverse[:, :-1] = x_a_inverse[:, 1:].clone()
        # x_a_inverse = self.press_layer(x_a_inverse)
        # x_a_inverse, _ = self.action_code_book(x_a_inverse, masks=attention_mask, use_cache=use_cache)
        # x_a_inverse = self.unpress_layer(x_a_inverse)

        # policy bc
        x_a_policy = self.transformer_policy.wte(input_ids) 
        attention_mask_xa = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x_a_policy, past_key_values_length
        )
        for block_a_policy in self.transformer_policy.h:
            layer_outputs = block_a_policy(
                x_a_policy,
                attention_mask=attention_mask_xa,
                position_ids=position_ids,
                past_key_value=past_key_values_policy,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x_a_policy = layer_outputs[0]
            if use_cache:
                next_decoder_cache_policy = layer_outputs[2 if output_attentions else 1]
        
        x_a_policy = self.transformer_policy.ln_f(x_a_policy)
        x_a_policy_logits = self.transformer_policy_layer(x_a_policy)

        x_a_policy, action_idx = self.action_code_book.forward_with_idx(
            x_a_policy_logits, 
            deterministic=self.deterministic, 
            epsilon_greedy=self.epsilon_greedy, 
            action_idx=action_idx
        )
        x_a_policy = self.unpress_layer(x_a_policy)

        self.set_action_info(action_idx=action_idx[:, -1:])

        # x = self.transformer.wte(input_ids)
        # mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), x, past_key_values_length
        # )
        # for num_block, block in enumerate(self.transformer.h):
        #     if num_block == self.enc_layer_num:
        #         x_a_policy = self.dynamics_norm(self.dynamics_layer(torch.cat([x, x_a_policy], dim=-1)))
        #         x_policy_s = x_a_policy + x

        #         # x = x_policy_s
        #         # use_cache to accelerate
        #         if use_cache:
        #             x= x_policy_s
        #         else:
        #             print("========Use ID ===================")
        #             x_a_inverse = self.dynamics_norm(self.dynamics_layer(torch.cat([x, x_a_inverse], dim=-1)))
        #             x_a_inverse_s = x_a_inverse + x
                    
        #             x = torch.cat([x_a_inverse_s, x_policy_s], dim=1)  # bs, 2*lens, dim
        #             # cos = torch.cat([cos, cos], dim=0)  # 2*lens, dim
        #             # sin = torch.cat([sin, sin], dim=0)  # 2*lens, dim
        #             if position_ids is not None:
        #                 position_ids = torch.cat([position_ids, position_ids], dim=-1)  # 2*lens
                        
        #             mask_sp = torch.cat([
        #                     torch.ones((seq_length, seq_length), dtype=torch.bool).tril(), 
        #                     torch.zeros((seq_length, seq_length), dtype=torch.bool)
        #                 ],dim=-1
        #             )
        #             mask_sp = mask_sp.unsqueeze(0).unsqueeze(0).to(x.device)

        #             mask_sa = torch.cat([
        #                 (torch.ones((seq_length, seq_length)).tril() - torch.eye(seq_length)).bool(), 
        #                 torch.eye(seq_length, dtype=torch.bool)
        #                 ], dim=-1
        #             )
        #             mask_sa = mask_sa.unsqueeze(0).unsqueeze(0).to(x.device)
        #             mask = torch.cat([mask_sp, mask_sa], dim=2)
                    
        #             if attention_mask is not None:
        #                 mask = mask.repeat(attention_mask.shape[0], 1, 1, 1)
        #                 pad_mask_ = torch.cat([attention_mask, attention_mask], dim=1)
        #                 mask = mask.permute(0, 1, 3, 2)
        #                 mask[pad_mask_.unsqueeze(1) == 0] = False
        #                 mask[:, :, torch.arange(seq_length * 2), torch.arange(seq_length * 2)] = True
        #                 mask = mask.permute(0, 1, 3, 2)
                    
        #     layer_outputs = block(
        #         x,
        #         attention_mask=mask,
        #         position_ids=position_ids,
        #         past_key_value=past_key_values_transformer,
        #         output_attentions=output_attentions,
        #         use_cache=use_cache,
        #     )
        #     x = layer_outputs[0]
        #     if use_cache:
        #         next_decoder_cache_transformer = layer_outputs[2 if output_attentions else 1]
        
        # if not use_cache:
        #     x = x[:, seq_length:]
        # x = self.transformer.ln_f(x)

        if use_cache:
            next_cache_policy = next_decoder_cache_policy.to_legacy_cache() if use_legacy_cache else next_decoder_cache_policy
            # next_cache_inverse_dynamic = next_decoder_cache_inverse_dynamic.to_legacy_cache() if use_legacy_cache else next_decoder_cache_inverse_dynamic
            # next_cache_transformer = next_decoder_cache_transformer.to_legacy_cache() if use_legacy_cache else next_decoder_cache_transformer

            next_cache = dict(
                policy_kv=next_cache_policy,
                # inverse_dynamics_kv=next_cache_inverse_dynamic,
                # transformer_kv=next_cache_transformer,
            )
        else:
            next_cache = None

        if not return_dict:
            return tuple(v for v in [x_a_policy_logits, next_cache, all_hidden_states, all_self_attns, action_idx] if v is not None)
        return IntentionModelOutputWithPast(
            last_hidden_state=x_a_policy_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            action_index=action_idx,
        )

    def forward_vqvae(
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
    ) -> Union[Tuple, IntentionModelOutputWithPast]:
        assert not use_cache 

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        past_key_values_policy = None
        past_key_values_inverse_dynamics = None
        past_key_values_transformer = None

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # inverse dynamics
        x_a_inverse = self.transformer_action_enc.wte(input_ids) 
        attention_mask_action_inverse = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x_a_inverse, past_key_values_length
        )
        for block_a_inverse in self.transformer_action_enc.h:
            layer_outputs = block_a_inverse(
                x_a_inverse,
                attention_mask=attention_mask_action_inverse,
                position_ids=position_ids,
                past_key_value=past_key_values_inverse_dynamics,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # print(layer_outputs, layer_outputs[0].shape)
            x_a_inverse = layer_outputs[0]

        x_a_inverse = self.transformer_action_enc.ln_f(x_a_inverse)
        x_a_inverse[:, :-1] = x_a_inverse[:, 1:].clone()
        x_a_inverse = self.press_layer(x_a_inverse)
        x_a_inverse, action_index = self.action_code_book(x_a_inverse, masks=attention_mask)
        x_a_inverse = self.unpress_layer(x_a_inverse)

        # policy bc
        x_a_policy = self.transformer_policy.wte(input_ids) 
        attention_mask_xa = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x_a_policy, past_key_values_length
        )
        for block_a_policy in self.transformer_policy.h:
            layer_outputs = block_a_policy(
                x_a_policy,
                attention_mask=attention_mask_xa,
                position_ids=position_ids,
                past_key_value=past_key_values_policy,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x_a_policy = layer_outputs[0]
        
        x_a_policy = self.transformer_policy.ln_f(x_a_policy)
        x_a_policy = self.transformer_policy_layer(x_a_policy)

        x_a_policy, _ = self.action_code_book.forward_with_idx(x_a_policy, deterministic=self.deterministic, epsilon_greedy=self.epsilon_greedy, action_idx=None)
        x_a_policy = self.unpress_layer(x_a_policy)

        x = self.transformer.wte(input_ids)
        mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x, past_key_values_length
        )
        for num_block, block in enumerate(self.transformer.h):
            if num_block == self.enc_layer_num:
                # sft step 1: trainning inverse dynamic and dynamic
                x_a_inverse = self.dynamics_norm(self.dynamics_layer(torch.cat([x, x_a_inverse], dim=-1)))
                x = x_a_inverse + x

            layer_outputs = block(
                x,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_values_transformer,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x = layer_outputs[0]
        
        x = self.transformer.ln_f(x)

        next_cache = None
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns, action_index] if v is not None)
        return IntentionModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            action_index=action_index
        )


if __name__ == "__main__":
    ACTOR_MODEL_PATH="/home/ubuntu/models/sft2-intention3B"
    TOKENIZER_PATH="/home/ubuntu/models/step2_reward-Llama2_7b-full_hh_rlhf-2023-12-23-17-41-01-1234/"
    # model = IntentionForCausalLM()
    tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    test = tokenizer(["Human: Can you tell me the stops on the B Line subway in Los Angeles? \nAssistant:", "In"], return_tensors="pt", pad_to_max_length=True)

    model_config = AutoConfig.from_pretrained(
        ACTOR_MODEL_PATH,
        # use_flash_attention_2=flash_attn,
        attn_implementation="eager",
    )
    model = IntentionForCausalLM.from_pretrained(ACTOR_MODEL_PATH, config=model_config, local_files_only=True)
    model.model.embed_tokens.load_state_dict(model.model.transformer_action_enc.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer_action_enc.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx)
    model.model.transformer_action_enc.wte.load_state_dict(model.model.embed_tokens.state_dict())

    model.model.embed_tokens = nn.Embedding(32000, model_config.hidden_size, model.model.padding_idx)
    model.model.embed_tokens.load_state_dict(model.model.transformer_policy.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer_policy.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx)
    model.model.transformer_policy.wte.load_state_dict(model.model.embed_tokens.state_dict())

    model.model.embed_tokens  = nn.Embedding(32000, model_config.hidden_size, model.model.padding_idx)
    model.model.embed_tokens.load_state_dict(model.model.transformer.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx)
    model.model.transformer.wte.load_state_dict(model.model.embed_tokens.state_dict())

    test_decode = model.generate(**test, max_length=50)
    print(test_decode)
    result = tokenizer.decode(test_decode[0])
    print(result)