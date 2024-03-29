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

### Parameter Indexing

```cpp
```

### Parameter Cache Coherence

```protobuf
// define gRPC message

service CacheCoherence {
    rpc GetParameter(Parameter) returns (Parameter) {}
}

message GetParameterRequest {
    int64 client_id = 1; // unique id for each client
}

message GetParameterResponse {
    bytes input = 1;
    bytes parameter = 2;
}
```

Each client will send a `GetParameterRequest` to the server, and the server will return a `GetParameterResponse` with the parameter.
The client will cache the parameter and use it for jobs.
The server will register and update the client id and its associated parameter.
Clients join and leave the server, and the server will update the client id and its associated parameter (zookeeper keep alive or on link disconnect, event based).

```python
# Server Implementation
class CacheCoherenceServicer(CacheCoherenceServicer):
    def __init__(self):
        pass

    def parameter_for_client(self, request):
        client_id = request.client_id
        ... # handle client cache here

    def GetParameter(self, request, context):
        client_id = request.client_id
        if is_registered(client_id):
            return GetParameterResponse(input=client_id, parameter=parameter_for_client(request))
        else:
            register_client(client_id)
            return GetParameterResponse(input=client_id, parameter=parameter_for_client(request))
```

### Model Parallelism Dispatcher

```python
# example of linear model 
model = nn.Sequential(
    nn.Linear(1024, 4096),
    ReLU(),
    nn.Linear(4096, 4096),
    ReLU(),
    nn.Linear(4096, 4096),
    ReLU(),
    nn.Linear(4096, 1024),
)

n_layers = 4
n_devices = 16

# each device has random delays
# TODO: implement whole row and column decomposition for tensor parallelism; more complex decomposition comes later

# K * N, N * M GeMM
def decompose_matrix(input, weight):
    tasks = []
    K, N = input.shape
    N, M = weight.shape

    for i in range(K):
        for j in range(M):
            tasks.append((i, j, input[i, :], weight[:, j])

    return tasks

# TODO: devices tasks tasks and execute them
def execute_tasks(tasks, devices):
    pass


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