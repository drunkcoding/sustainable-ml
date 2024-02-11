# Sustainable Machine Learning

## Installation

```bash
torch>=2.0
```

## Development Guide

- [ ] convert forward pass to use `torch.jit.script`
- [ ] convert backward pass to use `?` (research needed)

```python
def custom_forward_decorator(original_forward: Callable) -> Callable:
    @functools.wraps(original_forward)
    def wrapper(*args, **kwargs):
        return ... # replace with custom forward backends

    return wrapper

model = ... # Load model from HuggingFace's model hub

custom_backend = ... # replace with custom backend, initialize with model
custom_optimizer = ... # replace with custom optimizer, initialize with model parameters

for module in model.modules():
    if isinstance(module, nn.Linear):
        module.forward = custom_forward_decorator(module.forward)
        # activation function does not consume much time, so we don't need to parallelize it

# write our own backward pass, might need DeepSpeed code
custom_backend.backward()

# custom optimizer step, might need DeepSpeed code
custom_optimizer.step()

```

## Simulation Guide

```python
"""
Args:
    num_rounds: int. The number of rounds for the simulation.
    device_flops: List[int]. The maximum FLOPs for each device. Length is the
    number of devices.
    in_bandwidth: List[float]. The maximum ingress bandwidth for each device.
    out_bandwidth: List[float]. The maximum egress bandwidth for each device.
    model: str. The model to simulate, using HuggingFace's model hub.
    tp_depth: int. The depth of tensor parallelism.
    min_tp_size: int. The minimum size of the tensor parallelism. matrix less then this size will not be parallelized.
"""

model = ... # Load model from HuggingFace's model hub

param_flops = ... # calculate the number of FLOPs for Attention or MLP, using https://arxiv.org/pdf/2205.05198.pdf

param_size = ... # calculate the size of each nn.linear in bytes

STATE_NONE = -1
STATE_INGRESS = 0
STATE_COMPUTE = 1
STATE_EGRESS = 2

device_ts = [0 for _ in range(len(devices))]
device_state = [STATE_NONE for _ in range(len(devices))]
device_param = [None for _ in range(len(devices))]

for _ in range(num_rounds):
    # Simulate the training process

    # 1. Model forward/backward pass
    for param in model:
        md_list.extend(decompose_matrix(param, min_tp_size))

    # 2. generate tasks from parameters
    def take_task(md_list):
        pass

    while True:
        current_ts = max(device_ts)
        for device in devices:
            if device_ts[device] <= current_ts:
                if device_state[device] == STATE_NONE:
                    task = take_task(md_list)
                    device_state[device] = STATE_INGRESS
                    device_ts[device] = current_ts 
                    + task.param_size / in_bandwidth[device]
                    device_param[device] = task
                elif device_state[device] == STATE_INGRESS:
                    task = device_param[device]
                    device_state[device] = STATE_COMPUTE
                    device_ts[device] = current_ts 
                    + task.param_flops / device_flops[device]
                elif device_state[device] == STATE_COMPUTE:
                    task = device_param[device]
                    device_state[device] = STATE_EGRESS
                    device_ts[device] = current_ts 
                    + param_size / out_bandwidth[device]
                elif device_state[device] == STATE_EGRESS:
                    device_param[device] = None
                    device_state[device] = STATE_NONE
                
        if all_finished(model):
            break

    max_ts = max(device_ts)
    print(f"Round {_} finished at {max_ts} seconds")
```