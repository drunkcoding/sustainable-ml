import numpy as np
import transformers
import argparse
from transformers import AutoConfig

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def gemm_flops(m, n, k):
    return 2 * m * n * k


def allreduce_time(size, bw, dp):
    return size / bw * 2 * (dp - 1) / dp

def allgather_time(size, bw, n):
    return size / bw * (n - 1) / n

def reducescatter_time(size, bw, n):
    return size / bw * (n - 1) / n

def gossipscatter_time(size, bw, n):
    return size / bw * np.log2(n)

def comm_time(size, bw, pp):
    return size / bw * (pp - 1)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
parser.add_argument("--flops", type=float, default=3.0, help="FLOPS for the hardware (TFLOPS)")
parser.add_argument("--dtype", type=str, default="float", help="Data type (float or half)")
parser.add_argument("--num_devices", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--seq_len", type=int, default=512, help="sequence length")
parser.add_argument("--in_bw", type=float, default=100, help="ingress bandwidth (MB/s)")
parser.add_argument("--out_bw", type=float, default=5, help="egress bandwidth (MB/s)")
# parser.add_argument("--dp", type=int, default=1, help="number of data parallelism")
# parser.add_argument("--pp", type=int, default=1, help="number of pipeline parallelism")

args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model)

num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
ffn_size = config.ffn_dim

batch_size = args.batch_size
seq_len = args.seq_len
dtype_size = 4 if args.dtype == "float" else 2


# assert 4096*32 >= batch_size * seq_len, "batch_size * seq_len must be less than 4096*32"

def layer_flops(batch_size):
    # forward flops, backward needs double the flops
    K_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
    V_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
    Q_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
    O_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)

    attn_flops = gemm_flops(batch_size * seq_len, hidden_size, seq_len)
    attn_proj_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)

    fc1_flops = gemm_flops(batch_size * seq_len, ffn_size, hidden_size)
    fc2_flops = gemm_flops(batch_size * seq_len, hidden_size, ffn_size)

    layer_flops = (
        K_flops
        + V_flops
        + Q_flops
        + O_flops
        + attn_flops
        + attn_proj_flops
        + fc1_flops
        + fc2_flops
    )

    return layer_flops

Q_shape = (hidden_size, hidden_size)
K_shape = (hidden_size, hidden_size)
V_shape = (hidden_size, hidden_size)
O_shape = (hidden_size, hidden_size)

fc1_shape = (hidden_size, ffn_size)
fc2_shape = (ffn_size, hidden_size)

Q_size = np.prod(Q_shape) * dtype_size
K_size = np.prod(K_shape) * dtype_size
V_size = np.prod(V_shape) * dtype_size
O_size = np.prod(O_shape) * dtype_size

fc1_size = np.prod(fc1_shape) * dtype_size
fc2_size = np.prod(fc2_shape) * dtype_size



layer_size = Q_size + K_size + V_size + O_size + fc1_size + fc2_size
intermediate_size = batch_size * seq_len * hidden_size * dtype_size




# num_dp_stage = args.dp
# num_pp_stage = args.pp

# assert (
#     num_dp_stage * num_pp_stage == args.num_devices
# ), "DP * PP must equal the number of devices"

hybrid_latency_list = []
for num_dp_stage in range(1, args.num_devices + 1):
    for num_pp_stage in range(2, args.num_devices + 1):
        if num_dp_stage * num_pp_stage != args.num_devices or num_pp_stage > num_layers * 5 or num_dp_stage > batch_size:
            continue

        compute_time = num_layers * layer_flops(batch_size // num_dp_stage) / (args.flops * 1e12) * 3 # 3 = 1F + 2B
        pp_comm_time = comm_time(intermediate_size // num_dp_stage, args.out_bw * 1e6, num_pp_stage) + comm_time(num_layers * layer_size // num_pp_stage, args.out_bw * 1e6, num_pp_stage)   # 1F + 1B
        # fully_overlap_pp_comm_time = max(pp_comm_time, compute_time) - min(compute_time, pp_comm_time)
        # dp_allreduce_time = allreduce_time(layer_size * num_layers, args.out_bw * 1e6, num_dp_stage) 
        dp_allreduce_time = layer_size * num_layers * (num_dp_stage - 1) / (args.out_bw * 1e6 * num_dp_stage)
        # fully_overlap_allreduce_time = max(dp_allreduce_time, compute_time / 3 * 2) - min(dp_allreduce_time, compute_time / 3 * 2)
        # print(compute_time, batch_size, num_dp_stage, pp_comm_time, allredice_time, fully_overlap_pp_comm_time, fully_overlap_allreduce_time)
        total_latency = (
            # compute_time + fully_overlap_pp_comm_time + fully_overlap_allreduce_time
            compute_time + pp_comm_time + dp_allreduce_time
            # + num_layers * allreduce_time(layer_size, args.out_bw * 1e6, num_dp_stage)
            # + comm_time(intermediate_size // num_dp_stage, args.out_bw * 1e6, num_pp_stage) * 2 # 2 = 1F + 1B
        )

        hybrid_latency_list.append(
            {
                "num_dp_stage": num_dp_stage,
                "num_pp_stage": num_pp_stage,
                "total_latency": total_latency,
            }
        )
    

STATE_NONE = -1
STATE_INGRESS = 0
STATE_COMPUTE = 1
STATE_EGRESS = 2

# matrix shape of the attention mechanism


# save as csv
import pandas as pd

df = pd.DataFrame(hybrid_latency_list)
df.to_csv("hybrid_latency.csv")

# scatter plot of the latency
import matplotlib.pyplot as plt

dp_stage = [x["num_dp_stage"] for x in hybrid_latency_list]
pp_stage = [x["num_pp_stage"] for x in hybrid_latency_list]
total_latency = [int(x["total_latency"]) for x in hybrid_latency_list]

print("total_latency Hybrid", min(total_latency))

# # 2D scatter plot total_latency as labeled values alongside the points
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dp_stage, pp_stage, c=total_latency, cmap="viridis", s=100)
# for i, txt in enumerate(total_latency):
#     ax.annotate(txt, (dp_stage[i], pp_stage[i]))

# plt.xlabel("Data Parallelism Stage")
# plt.ylabel("Pipeline Parallelism Stage")

# plt.savefig("hybrid_latency.pdf")

num_tp_stages = min(args.num_devices, hidden_size)
num_pp_stages = args.num_devices // batch_size if args.num_devices >= batch_size else 1

# compute_latency = num_layers * layer_flops(batch_size // num_tp_stages) / (args.flops * 1e12) * 3  * (num_layers-1)*num_layers / 2 # 3 = 1F + 2B
# print("compute_latency", compute_latency)

total_latency = (
    num_layers * layer_flops(batch_size // num_tp_stages) / (args.flops * 1e12) * 3  # * (num_layers-1)*num_layers / 2 # 3 = 1F + 2B
    + num_layers * comm_time(layer_size // num_tp_stages, args.in_bw * 1e6, 2) * 2 # tensor shards to device, 1F + 1B
    + num_layers * comm_time(intermediate_size, args.in_bw * 1e6, 2) * 8 * 2 # send all inputs to device, 1F + 1B
    # + num_layers * allgather_time(intermediate_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) * 8 # all gather result for all ops
    + num_layers * reducescatter_time(intermediate_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) * 8 # get all inputs from device, 1F + 1B
    + num_layers * reducescatter_time(layer_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) # get gradients in B
    # + num_layers * comm_time(layer_size // num_tp_stages, args.in_bw * 1e6, 2) # send new parameter to device
    # + num_layers * comm_time(layer_size // num_tp_stages, args.out_bw * 1e6, 2) # send grad to server
)

print("total_latency w/o client cache", total_latency)
print("intermediate_size (MB)", intermediate_size / MB)
print("hidden_comm_time_out", comm_time(layer_size // num_tp_stages, args.in_bw * 1e6, num_pp_stages))
print("hidden_comm_time_in", comm_time(intermediate_size, args.in_bw * 1e6, 2))
print("hidden_comm_time_out", reducescatter_time(intermediate_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages))
print("hidden_comm_time_in", reducescatter_time(layer_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages))

total_latency = (
    num_layers * layer_flops(batch_size // num_tp_stages) / (args.flops * 1e12) * 3  # * (num_layers-1)*num_layers / 2 # 3 = 1F + 2B
    + num_layers * comm_time(layer_size // num_tp_stages, args.in_bw * 1e6, 2) # tensor shards to device, 1F, cache for backward
    + num_layers * comm_time(intermediate_size, args.in_bw * 1e6, 2) * 8 # send all inputs to device, 1F, cache for backward
    # + num_layers * allgather_time(intermediate_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) * 8 # all gather result for all ops
    + num_layers * reducescatter_time(intermediate_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) * 8 # get all inputs from device, 1F + 1B
    + num_layers * reducescatter_time(layer_size // num_tp_stages, args.out_bw * 1e6, num_tp_stages) # get gradients in B
    # + num_layers * comm_time(layer_size // num_tp_stages, args.in_bw * 1e6, 2) # send new parameter to device
    # + num_layers * comm_time(layer_size // num_tp_stages, args.out_bw * 1e6, 2) # send grad to server
)

print("total_latency w/ client cache", total_latency)