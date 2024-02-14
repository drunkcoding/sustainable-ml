import transformers
import tensor_parallel as tp
import torch
from torchviz import make_dot

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-6.7b")
model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")

model = model.half()  # <- half precision (optional, but recommended for large models)

model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])  # <- each GPU has half the weights

inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs, num_beams=5)
print(tokenizer.decode(outputs[0])) # A cat sat on my lap for a few minutes ...

loss = model(input_ids=inputs, labels=inputs).loss  # training works as usual

make_dot(loss).render("tp_oss", format="pdf")  # visualize the computation graph



