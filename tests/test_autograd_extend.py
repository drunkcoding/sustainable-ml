import time
import torch
from functools import partial
from torch.autograd import Function
from torchviz import make_dot
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import asyncio
import concurrent
import multiprocessing as mp
from edgeml.ops.op_builder.prefetch import BackendBuilder

gpu_unit = 0.1
num_gpus = torch.cuda.device_count()
num_actor_gpus = int(num_gpus / gpu_unit)
num_actor_gpus = int(2 ** (num_actor_gpus.bit_length() - 1))

def delegated_mm(A, B, gid):
    print("mm", gid, torch.cuda.memory_reserved(gid) / 1024**3, A.numel() * A.element_size() / 1024**3, B.numel() * B.element_size() / 1024**3)
    return A.matmul(B.to(f"cuda:{gid}")).to("cpu")


def backward_test(device):
    config = AutoConfig.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.bfloat16)
    decoder_layer = OPTDecoderLayer(config).to(device)

    hidden_size = config.hidden_size
    seq_len = 1024
    batch_size = 32

    hidden_states = torch.rand(batch_size, seq_len, hidden_size)
    hidden_states.requires_grad = False
    hidden_states = hidden_states.to(device)
    # attention_mask = torch.ones(batch_size, seq_len, device="cuda")
    # position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, seq_len)

    # get GPU free memory in GB
    print(torch.cuda.memory_reserved(0) / 1024**3)

    # output = decoder_layer(hidden_states, position_ids=position_ids)
    output = decoder_layer(hidden_states)
    # print("output", output)
    start_time = time.time()
    output[0].sum().backward()
    print("time", device, time.time() - start_time)

if __name__ == "__main__":
    mp.freeze_support()

    torch.multiprocessing.set_start_method('spawn')

    # pool = mp.Pool(num_actor_gpus)

    backend_lib = BackendBuilder().load()
    backend = backend_lib.matmul_threadpool(4)

    # @ray.remote(num_gpus=gpu_unit)
    # class GPUActor:
    #     async def mm(self, A, B):
    #         # print("I live in a pod with GPU access.")
    #         print("mm", A.shape, B.shape, A.device, B.device, torch.cuda.memory_reserved(0) / 1024**3)
    #         return A.to("cuda").matmul(B.to("cuda")).to("cpu")

    # print("num_gpus", num_gpus, "num_actor_gpus", num_actor_gpus)
    # round to lower 2^n

    if mp.parent_process() is None:
        backward_test("cuda:0")
    torch.cuda.empty_cache()

    print("num_gpus", num_gpus, "num_actor_gpus", num_actor_gpus)

    # gpu_actors = [GPUActor.remote() for _ in range(num_actor_gpus)]

    # pool_funcs = [partial(delegated_mm, gid=(i % num_gpus)) for i in range(num_actor_gpus)]

    class LinearFunction(Function):

        # Note that forward, setup_context, and backward are @staticmethods
        @staticmethod
        def forward(input, weight, bias=None):
            print("forward", input.shape, weight.shape, bias.shape if bias is not None else None)
            
            output = LinearFunction.submit_to_ray(input, weight)
            
            # output = input.matmul(weight)
            print("forward", torch.cuda.memory_reserved(0) / 1024**3, output.numel() * output.element_size() / 1024**3)
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

            
            # torch.cuda.empty_cache()
            return output
            # """
            # Implements a linear transformation for input tensors of any shape with weight tensors
            # also of any shape, using torch.mm for matrix multiplication. This function manually
            # handles the dimensions for compatibility with torch.mm.

            # Parameters:
            # - input: Input tensor of shape (*, in_features), where * denotes any number of preceding dimensions.
            # - weight: Weight tensor of shape (..., out_features, in_features).
            # - bias: Optional bias tensor. Its shape must be compatible with the output.

            # Returns:
            # - Output tensor.
            # """
            # print("forward", input.shape, weight.shape, bias.shape if bias is not None else None)
            # # Flatten input to 2D if necessary
            # original_input_shape = input.shape[:-1]
            # in_features = input.shape[-1]
            # input_2d = input.reshape(-1, in_features)

            # # Determine product of additional weight dimensions
            # additional_dims = weight.shape[:-2]
            # num_matrices = int(torch.prod(torch.tensor(additional_dims)).item())

            # # Reshape weight to treat additional dimensions as batches
            # weight_reshaped = weight.reshape(num_matrices, weight.shape[-2], weight.shape[-1])

            # # Initialize output list
            # output_list = []

            # # Perform matrix multiplication for each weight matrix
            # for i in range(num_matrices):
            #     if input_2d.shape[-1] == weight_reshaped[i].shape[-2]:
            #         output_2d = torch.mm(input_2d, weight_reshaped[i])
            #     else:
            #         output_2d = torch.mm(input_2d, weight_reshaped[i].t())
            #     output_list.append(output_2d)

            # # Stack and reshape output to match expected dimensions
            # output_stacked = torch.stack(output_list, dim=0)
            # output_shape = additional_dims + original_input_shape + (weight.shape[-2],)
            # output = output_stacked.reshape(output_shape)

            # # Add bias if provided
            # if bias is not None:
            #     bias_expanded = bias.view(additional_dims + (1,) * len(original_input_shape) + (bias.shape[-1],))
            #     output += bias_expanded

            # return output

        @staticmethod
        # inputs is a Tuple of all of the inputs passed to forward.
        # output is the output of the forward().
        def setup_context(ctx, inputs, output):
            input, weight, bias = inputs
            ctx.save_for_backward(input, weight, bias)

        # This function has only a single output, so it gets only one gradient
        @staticmethod
        def backward(ctx, grad_output):
            # This is a pattern that is very convenient - at the top of backward
            # unpack saved_tensors and initialize all gradients w.r.t. inputs to
            # None. Thanks to the fact that additional trailing Nones are
            # ignored, the return statement is simple even when the function has
            # optional inputs.
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            print("grad_output", grad_output.shape, input.shape, weight.shape, bias.shape if bias is not None else None)
            print("backward", torch.cuda.memory_reserved(0) / 1024**3, grad_output.numel() * grad_output.element_size() / 1024**3)

            # These needs_input_grad checks are optional and there only to
            # improve efficiency. If you want to make your code simpler, you can
            # skip them. Returning gradients for inputs that don't require it is
            # not an error.
            if ctx.needs_input_grad[0]:
                # grad_input = grad_output.mm(weight)
                # grad_input = ray.get(gpu_actors[0].mm.remote(grad_output.view(-1, grad_output.shape[-1]), weight.t()))
                grad_input_shape = grad_output.shape[:-1] + (weight.shape[-2],)
                grad_input = LinearFunction.submit_to_ray(grad_output.reshape(-1, grad_output.shape[-1]), weight.t())
                # grad_input = LinearFunction.normal_mm(grad_output.reshape(-1, grad_output.shape[-1]), weight.t())
                grad_input = grad_input.reshape(grad_input_shape)
            if ctx.needs_input_grad[1]:
                # grad_weight = grad_output.t().mm(input)
                # grad_weight = ray.get(gpu_actors[0].mm.remote(grad_output.t(), input)).t()
                # print("grad_output.t()", grad_output.t().shape, "input", input.shape)
                grad_weight_shape = (grad_output.shape[-1], ) + input.shape[-1:]
                grad_weight = LinearFunction.submit_to_ray(grad_output.reshape(-1, grad_output.shape[-1]).t(), input.reshape(-1, input.shape[-1]))
                # grad_weight = LinearFunction.normal_mm(grad_output.reshape(-1, grad_output.shape[-1]).t(), input.view(-1, input.shape[-1]))
                grad_weight = grad_weight.reshape(grad_weight_shape).t()
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)

            # ray.get([actor.mm.remote(grad_input, grad_weight, grad_bias) for actor in gpu_actors])

            return grad_input, grad_weight, grad_bias
        
        @staticmethod 
        def normal_mm(A, B):
            return A.to("cuda").matmul(B.to("cuda")).to("cpu")

        @staticmethod
        def submit_to_ray(A, B):
            # split B column-wise and send to each actor
            size_each_actor = B.shape[1] // num_actor_gpus

            print("submit_to_ray", A.shape, B.shape, size_each_actor)

            if size_each_actor == 0:
                B = B.split(1, dim=1)
                num_proc = 1
            else:
                B = B.split(size_each_actor, dim=1)  
                num_proc = len(B)

            B = [b.detach().contiguous() for i, b in enumerate(B)]
            A = A.detach().contiguous()
            # B = [b.to(f"cuda:{i % num_gpus}") for i, b in enumerate(B)]

            # one copy of A on all GPU
            # A_gpus = [A.to(f"cuda:{i}") for i in range(num_gpus)]

            outputs = [torch.zeros(1, device=f"cuda:{i % num_gpus}", requires_grad=False) for i in range(num_proc)]
            for i in range(num_proc):
                # print("enqueue", i, A_gpus[i % num_gpus].shape, B[i].shape, outputs[i].shape)
                # backend.enqueue(A_gpus[i % num_gpus], B[i], outputs[i], i % num_gpus)
                backend.enqueue(A, B[i], outputs[i], i % num_gpus)

            outputs = backend.wait_all()
            # outputs = [output.to("cpu") for output in outputs]
            # outputs = torch.cat(outputs, dim=-1)
            outputs.requires_grad = True

            # print(outputs)

            # global pool
            # outputs = pool.starmap(delegated_mm, [(A_gpus[i % num_gpus], B[i].detach(), i % num_gpus) for i in range(num_proc)])
            # outputs = torch.cat(outputs, dim=-1)

            # mp.freeze_support()
            # with mp.Pool(num_proc) as pool:
            #     outputs = pool.starmap(delegated_mm, [(A.detach(), B[i].detach(), i % num_gpus) for i in range(num_proc)])
            #     outputs = torch.cat(outputs, dim=-1)

            # async def async_get(actor, *args, **kwargs):
            #     return await actor.mm.remote(*args, **kwargs)

            # tasks = [async_get(actor, A, B[i]) for i, actor in enumerate(gpu_actors)]
            # print("submit_to_ray", A.shape, B[0].shape, size_each_actor)
            # asyncio.run(asyncio.wait(tasks))
            # print("submit_to_ray")
            # # outputs = torch.cat([ray.get(task.result()) for task in tasks], dim=-1)

            # refs = [actor.mm.remote(A, B[i]) for i, actor in enumerate(gpu_actors)]
            # futs = [ref.future() for ref in refs]
            # outputs = torch.cat([fut.result() for fut in concurrent.futures.as_completed(futs)], dim=-1)

            # async submit to each actor
            # outputs = [actor.mm.remote(A, B[i]) for i, actor in enumerate(gpu_actors)]
            # outputs = torch.cat(ray.get(outputs), dim=-1)
            print("submit_to_ray", outputs.shape, size_each_actor)
            # gather outputs
            return outputs
        

    linear = LinearFunction.apply
    # matmul = torch.matmul
    torch.matmul = LinearFunction.apply

    def forward_decorator(func):
        def wrapper(self, input):
            # print("forward_decorator", input.shape)
            return LinearFunction.apply(input, self.weight.t(), self.bias)
        return wrapper
    torch.nn.Linear.forward = forward_decorator(torch.nn.Linear.forward)

    # config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # decoder_layer = LlamaDecoderLayer(config, 0).to("cuda")

    # run code only on main process
    if mp.parent_process() is None:
        backward_test("cpu")

    exit()


# # from torch.autograd import gradcheck

# # # gradcheck takes a tuple of tensors as input, check if your gradient
# # # evaluated with these tensors are close enough to numerical
# # # approximations and returns True if they all verify this condition.
# # input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
# # test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
# # print(test)

# class DummyModule(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.fc1 = torch.nn.Linear(3, 2, bias=False)
#         self.fc2 = torch.nn.Linear(2, 2, bias=False)

#         self.fc1.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
#         self.fc2.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

#         self.m = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

#     def forward(self, input):
#         # print("forward", input.shape, self.fc1.weight.shape, self.fc2.weight.shape)
#         r1 = self.fc1(input)
#         # print("r1", r1.shape, r1)
#         r2 = self.fc2(r1)

#         r2 = torch.matmul(r2, self.m)
#         print("r2", r2.shape, r2)
#         return r2.sum()
    
# model = DummyModule()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 6.0]])

# start_time = time.time()
# # for _ in range(1):
# #     optimizer.zero_grad()
# #     yhat = model(input)
# #     # print("yhat", yhat)
# #     y = torch.tensor(1.0)
# #     loss = torch.pow(yhat - y, 2)
# #     loss.backward()
# #     optimizer.step()
# yhat = model(input)
# print("yhat", yhat)
# y = torch.tensor(1.0)
# loss = torch.pow(yhat - y, 2)
# loss.backward()
# print("time", time.time() - start_time)

# # make_dot(loss, show_attrs=True, show_saved=False, params=dict(model.named_parameters())).render("loss", format="pdf")

# # ray.shutdown()