import numpy as np
import transformers
import argparse
from transformers import AutoConfig


def gemm_flops(m, n, k):
    return 2 * m * n * k


def allreduce_time(size, bw, dp):
    return size / bw * 2 * (dp - 1) / dp


def comm_time(size, bw, pp):
    return size / bw * (pp - 1)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
parser.add_argument("--flops", type=float, default=3.0, help="FLOPS for the hardware (TFLOPS)")
parser.add_argument("--dtype", type=str, default="float", help="Data type (float or half)")
parser.add_argument("--num_devices", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
parser.add_argument("--in_bw", type=int, default=100, help="ingress bandwidth (MB/s)")
parser.add_argument("--out_bw", type=int, default=10, help="egress bandwidth (MB/s)")
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

# forward flops, backward needs double the flops
K_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
V_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
Q_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)
O_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)

attn_flops = gemm_flops(batch_size * seq_len, hidden_size, seq_len)
attn_proj_flops = gemm_flops(batch_size * seq_len, hidden_size, hidden_size)

fc1_flops = gemm_flops(batch_size * seq_len, ffn_size, hidden_size)
fc2_flops = gemm_flops(batch_size * seq_len, hidden_size, ffn_size)

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
        if num_dp_stage * num_pp_stage != args.num_devices or num_pp_stage > num_layers // 2:
            continue

        total_latency = (
            layer_flops / (args.flops * 1e9)
            + num_layers * allreduce_time(layer_size, args.out_bw * 1e6, num_dp_stage)
            + comm_time(intermediate_size, args.out_bw * 1e6, num_pp_stage)
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


# 2D scatter plot total_latency as labeled values alongside the points
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dp_stage, pp_stage, c=total_latency, cmap="viridis", s=100)
for i, txt in enumerate(total_latency):
    ax.annotate(txt, (dp_stage[i], pp_stage[i]))

plt.xlabel("Data Parallelism Stage")
plt.ylabel("Pipeline Parallelism Stage")

plt.savefig("hybrid_latency.pdf")