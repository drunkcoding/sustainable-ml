from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers import AutoConfig
import torch

# config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

hidden_size = config.hidden_size
ffn_size = config.intermediate_size
attn_head_size = config.num_attention_heads
kv_head_size = config.num_key_value_heads

print(hidden_size, ffn_size, attn_head_size, kv_head_size)

attn = MixtralAttention(config, 0)

print(attn)

batch_size = 64
seq_len = 1024

hidden_states = torch.rand(batch_size, seq_len, hidden_size)
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
output = attn(hidden_states, position_ids=position_ids)

# print(output.shape)