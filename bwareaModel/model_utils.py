# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from torch import nn
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
from huggingface_hub import snapshot_download
from bwareaModel.model_hf import IntentionForCausalLM


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in (
            "dropout",
            "attention_dropout",
            "hidden_dropout",
            "activation_dropout",
        ):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """Convert CausalLM model to calculate loss in fp32"""

    def causal_lm_forward(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        kwargs = (
            dict() if model.config.model_type == "llama" else dict(head_mask=head_mask)
        )
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss,) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward


def create_intention_model(
    model_name_or_path,
    tokenizer,
    dropout=None,
    flash_attn=False,
    dtype=None,
):
    is_transformer_2 = transformers.__version__.startswith("4.36")
    # model_config_path = "/home/ubuntu/models/tinyllama-1.1b/"
    if is_transformer_2:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            # use_flash_attention_2=flash_attn,
            attn_implementation="flash_attention_2" if flash_attn else "eager",
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=flash_attn,
            # attn_implementation="flash_attention_2" if flash_attn else "eager",
        )

    # #TODO wangpy: test generate with ID's action in step 0 to t-1
    # model_config.use_cache = False

    configure_dropout(model_config, dropout)
    if hasattr(model_config, "sliding_window"):
        if model_config.sliding_window is None:
            model_config.sliding_window = 4096

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration

    model = IntentionForCausalLM.from_pretrained(
        model_name_or_path, 
        config=model_config,
        # attn_implementation="flash_attention_2" if flash_attn else "eager",
        local_files_only=True,
        torch_dtype=dtype,
    )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.pad_token_id = model.config.eos_token_id

    # model.model.embed_tokens.load_state_dict(model.model.transformer_action_enc.wte.state_dict())
    # resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    # model.resize_token_embeddings(
    #     resized_token_dim
    # )  # make the vocab size multiple of 8
    if dtype is not None:
        model.model.embed_tokens = nn.Embedding(model_config.vocab_size, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.embed_tokens.load_state_dict(model.model.transformer_action_enc.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer_action_enc.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.transformer_action_enc.wte.load_state_dict(model.model.embed_tokens.state_dict())

    model.model.embed_tokens = nn.Embedding(model_config.vocab_size, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.embed_tokens.load_state_dict(model.model.transformer_policy.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer_policy.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.transformer_policy.wte.load_state_dict(model.model.embed_tokens.state_dict())

    model.model.embed_tokens = nn.Embedding(model_config.vocab_size, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.embed_tokens.load_state_dict(model.model.transformer.wte.state_dict())
    resized_token_dim = int(8 * math.ceil(len(tokenizer) / 8.0))
    model.resize_token_embeddings(
        resized_token_dim
    )  # make the vocab size multiple of 8
    model.model.transformer.wte = nn.Embedding(resized_token_dim, model_config.hidden_size, model.model.padding_idx, dtype=dtype)
    model.model.transformer.wte.load_state_dict(model.model.embed_tokens.state_dict())

    return model

def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    rlhf_training=False,
    dropout=None,
    flash_attn=False,
    dtype=None,
):
    is_transformer_2 = transformers.__version__.startswith("4.36")
    if is_transformer_2:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            # use_flash_attention_2=flash_attn,
            attn_implementation="flash_attention_2" if flash_attn else "eager",
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=flash_attn,
            # attn_implementation="flash_attention_2" if flash_attn else "eager",
        )

    configure_dropout(model_config, dropout)
    if hasattr(model_config, "sliding_window"):
        if model_config.sliding_window is None:
            model_config.sliding_window = 4096

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if rlhf_training:
        # the weight loading is handled by create critic model
        if flash_attn and not is_transformer_2:
            model_config._flash_attn_2_enabled = True
        model = model_class.from_config(model_config)
    else:
        if is_transformer_2:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                # use_flash_attention_2=flash_attn,
                attn_implementation="flash_attention_2" if flash_attn else "eager",
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                use_flash_attention_2=flash_attn,
                # attn_implementation="flash_attention_2" if flash_attn else "eager",
                torch_dtype=dtype,
                local_files_only=True,
            )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    print(f"model.resize_token_embeddings: {int(8 * math.ceil(len(tokenizer) / 8.0))}")
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    return model