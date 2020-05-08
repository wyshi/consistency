import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from GPTModel1 import GPT2LMHeadModel_modified
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
import config as cfg

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.eval()

model_A_states, model_B_states = torch.load()#torch.load(cfg.old_medium_model_dir)
model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
model.load_state_dict(model_A_states, strict=False)

input_ids = torch.tensor(tokenizer.encode("They are an international non-governmental organization that promotes children's rights, provides relief and helps support children in need.  In 2017 alone, 1,000 children were reportedly killed or injured in intensifying violence.  In the first two months of 2018 alone, 1,000 children were reportedly killed or injured in intensifying violence. ")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]

PAD_TOKEN = tokenizer.encoder["<|endoftext|>"]

mask = input_ids.ne(PAD_TOKEN).float()
logits = logits[:, :-1].contiguous()
target = input_ids[:, 1:].contiguous()
mask = mask[:, 1:].contiguous()

criterion = SequenceCrossEntropyLoss()

sequences_logprobs = - criterion(logits, target, mask)
old_logprobs = sequences_logprobs.sum(1)
old_logprobs


buffer = 
for data in train_data:
    for s in data:
        if "researchers gather the for the researchers the researchers collect" in s:
            print(data)