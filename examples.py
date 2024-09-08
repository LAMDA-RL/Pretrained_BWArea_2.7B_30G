import torch
from bwareaModel.model_utils import create_intention_model
from bwareaModel.tokenizer import load_hf_tokenizer
# load tokenizer
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30B/",  # model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
# load model
model = create_intention_model(
    "../intention_pretrained_2.7B_30B/",  # model path
    tokenizer=tokenizer,
    dtype=torch.bfloat16
)

# using world model under given actions
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
fixed_action_idx = 2
outputs = model(input_ids=input_ids, attention_mask=attention_mask, action_idx=fixed_action_idx)
logits_next = outputs.logits[:, -1]
idx = logits_next.argmax(dim=1, keepdim=True)
output_ids = torch.cat([input_ids, idx], dim=-1).long().squeeze(0)
examples_out = tokenizer.decode(output_ids)
print(examples_out, "(fixed action idx = {})".format(fixed_action_idx))

# <s> I like eating! (fixed action idx = 0)
# <s> I like eating well (fixed action idx = 1)
# <s> I like eating n (fixed action idx = 2)
# <s> I like eating raw (fixed action idx = 3)
# <s> I like eating a (fixed action idx = 4)
# <s> I like eating correctly (fixed action idx = 5)
# <s> I like eating at (fixed action idx = 6)
# <s> I like eating them (fixed action idx = 7)
# <s> I like eating lots (fixed action idx = 8)
# <s> I like eating or (fixed action idx = 9)
# <s> I like eating out (fixed action idx = 10)
# <s> I like eating it (fixed action idx = 11)
# <s> I like eating in (fixed action idx = 12)
# <s> I like eating this (fixed action idx = 13)
# <s> I like eating pot (fixed action idx = 14)
# <s> I like eating bread (fixed action idx = 15)
# <s> I like eating my (fixed action idx = 16)
# <s> I like eating meat (fixed action idx = 17)
# <s> I like eating the (fixed action idx = 18)
# <s> I like eating car (fixed action idx = 19)

# or generate under tensor action:
fixed_action_idx = torch.randint(0, 64, size=input_ids.shape).long()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, action_idx=fixed_action_idx)
logits_next = outputs.logits[:, -1]
idx = logits_next.argmax(dim=1, keepdim=True)
output_ids = torch.cat([input_ids, idx], dim=-1).long().squeeze(0)
examples_out = tokenizer.decode(output_ids)
print(examples_out)

# <s> I like eating fresh


# using inverse dynamics model for language comprehension
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model.forward_inverse(input_ids=input_ids, attention_mask=attention_mask)
action_idx = outputs.action_index[:, :-1]
print(action_idx.shape, action_idx)

# torch.Size([1, 4]) tensor([[45, 45, 45, 45]])  # means that the sentence "I like eating mainly using action no.45"


# using policy model to select actions
model.set_action_sampling(greedy=False, temp=2.0)  # greedy=True for determinitic action, temp for temperature of action sampling
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model.forward_policy(input_ids=input_ids, attention_mask=attention_mask)

# get action logits
action_logits = outputs.logits[:, -1]

# get action index
action_index = outputs.action_index[:, -1]
print(action_index)


# using policy model for selecting actions and generating language with world model
# model = model.to("cuda:0")
model.set_action_sampling(greedy=False, temp=2.0)  # greedy=True for determinitic action, temp for temperature of action sampling
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# get selected action index
action_index = outputs.action_index

# get language logits by world model
logits = outputs.logits

# get embeddings
embeddings = outputs.last_hidden_state


### generate
# load tokenizer
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30B/",  # model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
# load model
model = create_intention_model(
    "../intention_pretrained_2.7B_30B/",  # model path
    tokenizer=tokenizer,
    dtype=torch.bfloat16
)

examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
batch_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}

with torch.no_grad():
    outputs = model.generate(
        **batch_inputs,
        max_new_tokens=10,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_p=1.0,
        temperature=0.8,
        top_k=1,
    )
    outputs = outputs.squeeze(0)
    examples_output = tokenizer.decode(outputs, skip_special_tokens=True)
    print(examples_output)

# I like eating something soothing and helping to tone my body and


# A Simple Example of Policy Training
# load tokenizer
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30B/",  # model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
# load model
model = create_intention_model(
    "../intention_pretrained_2.7B_30B/",  # model path
    tokenizer=tokenizer,
    dtype=torch.bfloat16
)

import torch.nn as nn
import torch.nn.functional as F
def mark_only_param_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if bias not in n:
            p.requires_grad = False
mark_only_param_as_trainable(model, bias="policy")
trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer  = torch.optim.AdamW(trainable_params, lr=1e-4)
def reward_function(seq):
    return torch.randn(seq.shape[0], 1)

batch_inputs = {
    "input_ids": torch.randint(10, 30000, size=(4, 16)).long(),
    "attention_mask": torch.ones((4, 16)).long(),
}
prompt_length = batch_inputs['input_ids'].shape[1]

# sampling
model.reset_action_info()
outputs = model.generate(
    **batch_inputs,
    max_new_tokens=10,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    top_p=1.0,
    temperature=0.8,
    top_k=1,
)
# reward labeling
reward = reward_function(outputs)

# get action index
acction_info = model.get_action_info()
action_idx = torch.cat(acction_info["action_idx"], dim=1)
model.reset_action_info()

# get action logits
outputs_mask = torch.ones_like(outputs).long()
outputs_mask[outputs_mask == tokenizer.pad_token_id] = 0
outputs = model.forward_policy(input_ids=outputs, attention_mask=outputs_mask)
action_logits = outputs.logits[:, prompt_length:]

# compute loss with reward, action_index and action_logits
action_mask = outputs_mask[:, prompt_length:]
action_log_probs = torch.log(F.softmax(action_logits, dim=-1))
action_log_probs = action_log_probs.gather(index=action_idx.unsqueeze(-1), dim=-1).squeeze(-1)

print(reward.shape, action_log_probs.shape, action_mask.shape)
loss = - (reward * action_log_probs * action_mask).sum() / action_mask.sum()
print(loss.item())

# optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()



