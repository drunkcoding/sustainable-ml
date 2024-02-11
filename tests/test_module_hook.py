import torch
import torch.nn as nn

def pre_forward_hook(module, args):
    print("pre_forward_hook")
    print(module)
    print(args)
    # print(kwargs)

def post_forward_hook(module, input, output):
    print("post_forward_hook")
    print(module)
    print(input)
    print(output)

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
        module.register_forward_pre_hook(pre_forward_hook)
        module.register_forward_hook(post_forward_hook)

        module.register_full_backward_pre_hook(pre_backward_hook)
        # module.register_full_backward_hook(post_backward_hook)
        module.register_backward_hook(backward_hook)

    return model



if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    print(model)

    # random initialization model paramters
    for param in model.parameters():
        param.data = torch.rand_like(param)

    rand_input = torch.rand(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # register hooks
    model = register_hooks(model)

    # train the model
    for _ in range(10):
        optimizer.zero_grad()
        output = model(rand_input)
        loss = torch.nn.functional.mse_loss(output, torch.ones(10, 1))
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

    # save the model
    # torch.save(model.state_dict(), "model.pth")