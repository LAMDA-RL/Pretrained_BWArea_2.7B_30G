# Pre-trained Models of BWArea Model

This repository provides the pretrained models for the paper "BWArea Model: Learning World Model, Inverse Dynamics, and Policy for Controllable Language Generation". A BWArea model is a complex system that consists of three main components:
* Language World Model: This model has an action space of 64 actions and generates language based on these actions.
* Inverse Dynamics Model: This model derives actions from input sentences, essentially "understanding" the language.
* Policy Model: This model was trained to output actions that simulate the language data used in training.

The BWArea model can simulate a language model by using the inverse dynamics model to understand (i.e., derive actions from) any given prompt, and then generate language by inputting actions from either the inverse dynamics model or the policy model.
However, the BWArea model is capable of more than just language generation. It is possible to train a custom policy to maximize any reward function, resulting in a task-specific policy model. The reward function can be manually designed, allowing for the creation of policies that accomplish various tasks such as negotiation, persuasion, playing text-based games, and more.

## Install

```
pip install -r requirements.txt
```

## Usage

BWArea Model contains three parts: Lanugage World Model (1.1B), Inverse Dynamics Model (0.5B) and Policy Model (1.1B), 2.7B in total. Each module can be utilized seperately or combined for distinguished objective.

### Loading Model and Tokenizer

```python
import torch
from bwareaModel.model_utils import create_intention_model
from bwareaModel.tokenizer import load_hf_tokenizer
# load tokenizer
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30G",  # your model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
# load model
model = create_intention_model(
    "../intention_pretrained_2.7B_30G",  # your model path
    tokenizer=tokenizer,
    dtype=torch.bfloat16
)      
```

### Use Language World Model for Language Generation

```python
# The language world model take actions as input and generate the next token.
# In this example, you can try different actions and see how the language world model generates
examples = "I like eating" # this is the prompt that is to be understood by the inverse dynamics model
fixed_action_idx = 2  # choose your action between 0 to 63
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, action_idx=fixed_action_idx)
logits_next = outputs.logits[:, -1]
idx = logits_next.argmax(dim=1, keepdim=True)
output_ids = torch.cat([input_ids, idx], dim=-1).long().squeeze(0)
examples_out = tokenizer.decode(output_ids)
print(examples_out, "(fixed action idx = {})".format(fixed_action_idx))

# generation by some different actions. Not that these outputs are not random tokens, but each has a certain semantic meaning.

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
```

#### or generate under tensor actions:

```python
fixed_action_idx = torch.randint(0, 64, size=input_ids.shape).long()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, action_idx=fixed_action_idx)
logits_next = outputs.logits[:, -1]
idx = logits_next.argmax(dim=1, keepdim=True)
output_ids = torch.cat([input_ids, idx], dim=-1).long().squeeze(0)
examples_out = tokenizer.decode(output_ids)
print(examples_out)

# <s> I like eating fresh
```

### Use Inverse Dynamics Model for Language Comprehension

```python
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model.forward_inverse(input_ids=input_ids, attention_mask=attention_mask)
action_idx = outputs.action_index[:, :-1]
print(action_idx.shape, action_idx)

# print outputs
# torch.Size([1, 4]) tensor([[45, 45, 45, 45]])  
# means that the sentence "I like eating" mainly using action no.45
  
```

### Use Policy Model to Select Actions

```python
# The policy model was pretrained according to the training data.
# This example shows the actions of the pre-trained policy
model.set_action_sampling(greedy=False, temp=2.0)  # greedy=True for deterministic action, temp for temperature of action sampling
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
```

### Use Policy Model for Generating Language with World Model

```python
# This example shows the intermediate variable of using the pretrained policy model in the language world model.
model.set_action_sampling(greedy=False, temp=2.0)  # greedy=True for determinitic action, temp for temperature of action sampling
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# get action index
action_index = outputs.action_index

# get logits
logits = outputs.logits

# get embeddings
embeddings = outputs.last_hidden_state
print(logits.shape, embeddings.shape)   
```

### Sentence Generation

```python
# This example uses the BWArea as a common LLM for language generation.
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
```
## Policy Model Training

Since BWArea model treats the language generation as a decision tasks on the language world model and a certain reward (human intention or specific tasks), one of the advantage of BWArea Model is that we can only optimize the policy model to align a certain human intention or tasks.

### A Simple Example of Policy Training (Using Reinforce)

```python
# This example trains the policy using your own reward function.
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

# define your reward function
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
```

## Acknowledgements

- (https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)


