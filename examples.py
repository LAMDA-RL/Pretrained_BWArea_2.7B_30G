import torch
from bwareaModel.model_utils import create_intention_model
from bwareaModel.tokenizer import load_hf_tokenizer
# load tokenizer
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30G",  # model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
# load model
model = create_intention_model(
    "../intention_pretrained_2.7B_30G",  # model path
    tokenizer=tokenizer,
    dtype=torch.bfloat16
)

# forward
# model = model.to("cuda:0")
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

### generate
tokenizer = load_hf_tokenizer(
    "../intention_pretrained_2.7B_30G",  # model path
    fast_tokenizer=True,
    add_special_tokens=None,
)
model = create_intention_model(
    "../intention_pretrained_2.7B_30G",  # model path
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






