# Pre-trained Models of BWArea Model

The pre-trained model of paper "BWArea Model: Learning World Model, Inverse Dynamics, and Policy for Controllable Language Generation". The model initially trained on 30G tokens for a verification. 

### Model

https://huggingface.co/jiacx0229/intention_pretrained_2.7B_30G

### Install

```
pip install -r requirements.txt
```

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

### Single Step output

```python
model.set_action_sampling(greedy=False, temp=2.0)  # greedy=True for determinitic action, temp for temperature of action sampling
examples = "I like eating"
encodes = tokenizer.encode(examples)
input_ids = torch.LongTensor(encodes).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# get logits
logits = outputs.logits

# get embeddings
embeddings = outputs.last_hidden_state
print(logits.shape, embeddings.shape)   
```

### Generation

```python
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30G",  # your model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
model = create_intention_model(
    "../intention_pretrained_2.7B_30G",  # your model path
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
```

### Acknowledgements

- (https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)


