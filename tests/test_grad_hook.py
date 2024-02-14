import functools
import inspect
import torch
import torch.nn as nn
from torchviz import make_dot

def pre_backward_hook(module, grad_input):
    print("pre_backward_hook")
    print("module:", module)
    if isinstance(module, nn.Linear):
        print(module.weight)
    print("grad_input:", grad_input)

def post_backward_hook(module, grad_input, grad_output):
    print("post_backward_hook")
    print("module:",module)
    print("grad_input:", grad_input)
    print("grad_output:", grad_output)

def backward_hook(module, grad_input, grad_output):
    print("backward_hook")
    print("module:",module)
    print("grad_input:", grad_input)
    print("grad_output:", grad_output)

def register_hooks(model):
    for name, module in model.named_children():
        module.register_full_backward_pre_hook(pre_backward_hook)
        # module.register_full_backward_hook(post_backward_hook)
        module.register_backward_hook(backward_hook)

    model.register_full_backward_pre_hook(pre_backward_hook)
    # model.register_full_backward_hook(post_backward_hook)
    model.register_backward_hook(backward_hook)
    return model

class DummyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(3, 2, bias=False)
        self.fc2 = torch.nn.Linear(2, 2, bias=False)

        self.fc1.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.fc2.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def forward(self, input):
        print("forward", input.shape, self.fc1.weight.shape, self.fc2.weight.shape)
        r1 = self.fc1(input)
        print("r1", r1.shape, r1)
        r2 = self.fc2(r1)
        return r2.sum()

model = DummyModule()

register_hooks(model)

input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 6.0]])
print("input", input)   

def custom_pack_hook(input):
    print("pack", input)
    return input

def custom_unpack_hook(input):
    print("unpack", input)
    return input

with torch.autograd.graph.saved_tensors_hooks(custom_pack_hook, custom_unpack_hook):
    yhat = model(input)
    print("yhat", yhat)
    y = torch.tensor(1.0)
    loss = torch.pow(yhat - y, 2)

def grad_fn_wrapper(grad_fn):
    @functools.wraps(grad_fn)
    def wrapper(*args, **kwargs):
        print("grad_fn_wrapper", args, kwargs)
        return grad_fn(*args, **kwargs)
# loss.grad_fn = grad_fn_wrapper(loss.grad_fn)

# loss.backward()

make_dot(loss, show_attrs=True, show_saved=False, params=dict(model.named_parameters())).render("loss", format="pdf")

# print("loss", loss)
# # print("loss", dir(loss.grad_fn), loss.grad_fn.next_functions, loss.grad_fn.name(), loss.grad_fn.metadata)
# # print("model.x", model.x)
# print('Grad fc1', model.fc1.weight.grad)
# print('Grad fc2', model.fc2.weight.grad)

# print(dir(loss.grad_fn.__init__))
# sig = inspect.signature(loss.grad_fn.__init__)
# parameters = sig.parameters

# # Listing argument names
# for name, param in parameters.items():
#     print(name)
# print("model.fc1.grad_fn", loss.grad_fn.__code__.co_varnames)

# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# # optimizer.step()

print("model.fc1", model.fc1.weight)
print("model.fc2", model.fc2.weight)
# print('Grad', model.x.grad)


# x = torch.arange(5, dtype=torch.float32).requires_grad_(True)
# out = torch.pow(x, x)
# print(out)
# # tensor([0.4651, 1.1575], grad_fn=<PowBackward0>)

# out.sum().backward(torch.tensor(2.0))
# print(x.grad)
