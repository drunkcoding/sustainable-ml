# training facebook/opt-6.7b using batch size 64 and sequence length 512

from transformers import AutoConfig
from transformers import OPTForCausalLM
import torch
import time

model_name = "facebook/opt-6.7b"
config = AutoConfig.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name, config=config)


input_ids = torch.randint(0, 1000, (64, 512))
labels = torch.randint(0, 1000, (64, 1))
attention_mask = torch.ones(64, 512)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for _i in range(10):
    start = time.time()
    optimizer.zero_grad()
    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    loss.backward()
    optimizer.step()
    print(f"Time taken for iteration {_i}: {time.time() - start:.2f} seconds")
