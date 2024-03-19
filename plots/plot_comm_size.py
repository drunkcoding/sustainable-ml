import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams.update({"font.size": 24})

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

SEQ_LENGTH = 1024
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 64
NUM_MICRO_BATCHES = BATCH_SIZE // MICRO_BATCH_SIZE

HIDDEN_SIZE = 7168
FFN_SIZE = 28672
NUM_LAYER = 48
DTYPE_SIZE = 2

LAYER_SIZE = (3 * HIDDEN_SIZE**2 + 2 * HIDDEN_SIZE * FFN_SIZE) * DTYPE_SIZE

BANDWIDTH_IN = 100 * MB
BANDWIDTH_OUT = 20 * MB

pp_rank_list = [2, 4, 8, 12, 24, 36, 48]
pp_forward_comm_size = [
    SEQ_LENGTH * BATCH_SIZE * HIDDEN_SIZE
    + SEQ_LENGTH * MICRO_BATCH_SIZE * HIDDEN_SIZE * (pp_rank - 1)
    for pp_rank in pp_rank_list
]
pp_backward_comm_size = [
    NUM_MICRO_BATCHES * LAYER_SIZE * NUM_LAYER
    + LAYER_SIZE * NUM_LAYER / pp_rank * (pp_rank - 1)
    for pp_rank in pp_rank_list
]
pp_forward_comm_latency = np.array(pp_forward_comm_size) / BANDWIDTH_OUT
pp_backward_comm_latency = np.array(pp_backward_comm_size) / BANDWIDTH_OUT
pp_comm_latency = pp_forward_comm_latency + pp_backward_comm_latency

pp_total_forward_comm_size = [
    SEQ_LENGTH * BATCH_SIZE * HIDDEN_SIZE * (rank - 1)
    for rank in pp_rank_list
]
pp_total_backward_comm_size = [
    LAYER_SIZE * NUM_LAYER * (rank - 1)
    for rank in pp_rank_list
]

pp_comm_size = np.array(pp_total_forward_comm_size) + np.array(pp_total_backward_comm_size)
pp_comm_size = pp_comm_size / GB

dp_rank_list = [1, 2, 4, 8, 12, 24]
hybrid_comm_latency = [
    pp_forward_comm_latency[-1] / dp_rank
    + pp_backward_comm_latency[-1]
    + LAYER_SIZE * NUM_LAYER * (dp_rank - 1) / (BANDWIDTH_OUT * pp_rank_list[-1])
    for i, dp_rank in enumerate(dp_rank_list)
]
hybrid_comm_size = [
    pp_forward_comm_size[-1] / dp_rank
    + pp_backward_comm_size[-1]
    + LAYER_SIZE * NUM_LAYER * (dp_rank - 1) / pp_rank_list[-1]
    for i, dp_rank in enumerate(dp_rank_list)
]
hybrid_total_comm_size = [  
    pp_comm_size[-1] * rank + LAYER_SIZE * NUM_LAYER * (rank - 1)
    for rank in dp_rank_list
]
hybrid_comm_size = np.array(hybrid_comm_size) / GB
hybrid_tp_comm_size = [
    pp_comm_size[-1] * rank
    for rank in dp_rank_list
]
hybrid_mp_comm_size = [
    pp_comm_size[-1] * rank
    + SEQ_LENGTH * BATCH_SIZE * HIDDEN_SIZE * NUM_LAYER * (rank - 1) / pp_rank_list[-1]
    for rank in dp_rank_list
]
# hybrid_tp_comm_size = np.array(hybrid_tp_comm_size) / GB

# hybrid_tp_forward_comm_latency = [
#     SEQ_LENGTH * BATCH_SIZE * HIDDEN_SIZE / rank
#     + SEQ_LENGTH * MICRO_BATCH_SIZE * HIDDEN_SIZE * (pp_rank_list[-1] - 1)
#     for rank in dp_rank_list
# ]
# pp_backward_comm_latency = [
#     NUM_MICRO_BATCHES * LAYER_SIZE * NUM_LAYER
#     + LAYER_SIZE * NUM_LAYER / pp_rank * (pp_rank - 1)
#     for pp_rank in pp_rank_list
# ]
# pp_forward_comm_latency = np.array(pp_forward_comm_latency) / BANDWIDTH_OUT
# pp_backward_comm_latency = np.array(pp_backward_comm_latency) / BANDWIDTH_OUT
# pp_comm_latency = pp_forward_comm_latency + pp_backward_comm_latency

hybrid_tp_comm_latency = [ pp_forward_comm_latency[-1] + pp_backward_comm_latency[-1] for rank in dp_rank_list]
# hybrid_tp_comm_latency = [
#     # pp_forward_comm_latency[-1] + pp_backward_comm_latency[-1] +
#     pp_forward_comm_latency[-1] / pp_rank_list[-1] * (dp_rank - 1) + pp_backward_comm_latency[-1] / pp_rank_list[-1] * (dp_rank - 1)
#     # + BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * NUM_LAYER * 5 / (BANDWIDTH_OUT * dp_rank) * (dp_rank - 1)
#     for i, dp_rank in enumerate(dp_rank_list)
# ]

hybrid_rank_list = [rank * pp_rank_list[-1] for rank in dp_rank_list]

tp_rank_list = pp_rank_list + hybrid_rank_list
tp_forward_comm_size = [
    SEQ_LENGTH * BATCH_SIZE * HIDDEN_SIZE * NUM_LAYER * 3
    + FFN_SIZE * HIDDEN_SIZE * NUM_LAYER * 2
    + BATCH_SIZE * SEQ_LENGTH**2 * NUM_LAYER
    for rank in tp_rank_list
]
tp_backward_comm_size = [LAYER_SIZE * NUM_LAYER for rank in tp_rank_list]


tp_comm_latency = (
    np.array(tp_forward_comm_size) / (BANDWIDTH_OUT * np.array(tp_rank_list))
    + np.array(tp_forward_comm_size) / BANDWIDTH_IN
    + np.array(tp_backward_comm_size) / BANDWIDTH_IN
    + np.array(tp_backward_comm_size) / (BANDWIDTH_OUT * np.array(tp_rank_list))
)
tp_comm_size = np.array(tp_forward_comm_size) + np.array(tp_backward_comm_size)
tp_comm_size = tp_comm_size / GB * (1 + np.array(tp_rank_list))
# tp_comm_latency = tp_forward_comm_latency + np.array(tp_backward_comm_latency) / (
#     BANDWIDTH_OUT * np.array(tp_rank_list)
# )


plt.figure(figsize=(13, 8))
plt.plot(pp_rank_list, pp_comm_latency, label="PP Latency", linewidth=3)
plt.plot(
    hybrid_rank_list,
    hybrid_comm_latency,
    label="DP+PP Latency",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    hybrid_rank_list,
    hybrid_tp_comm_latency,
    label="TP+PP TP Latency",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    tp_rank_list, tp_comm_latency, label="TP Latency", linestyle=":", linewidth=3
)
plt.xlabel("Number of PP ranks")
plt.ylabel("Latency (s)")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.savefig("plots/comm_latency.png", bbox_inches="tight")
plt.close()


plt.figure(figsize=(13, 8))
plt.plot(pp_rank_list, pp_comm_size, label="PP", linewidth=3)
plt.plot(
    hybrid_rank_list,
    hybrid_total_comm_size,
    label="DP+PP",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    hybrid_rank_list,
    hybrid_tp_comm_size,
    label="TP+PP (Column)",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    hybrid_rank_list,
    hybrid_mp_comm_size,
    label="TP+PP (Row)",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    tp_rank_list, tp_comm_size, label="TP", linestyle=":", linewidth=3
)
plt.xlabel("Number of Devices")
plt.ylabel("Size (GB)")
# plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("plots/comm_size.png", bbox_inches="tight")
plt.close()


plt.figure(figsize=(13, 8))
plt.plot(pp_rank_list, [BANDWIDTH_OUT / GB * rank for rank in pp_rank_list], label="PP BW", linewidth=3)
plt.plot(
    hybrid_rank_list,
    [BANDWIDTH_OUT / GB * rank for rank in hybrid_rank_list],
    label="DP+PP BW",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    hybrid_rank_list,
    [BANDWIDTH_OUT / GB * rank for rank in hybrid_rank_list],
    label="TP+PP BW",
    linestyle="--",
    linewidth=3,
)
plt.plot(
    tp_rank_list, [1 / ( 1/ (BANDWIDTH_OUT * rank) + 1 / BANDWIDTH_IN) / GB * rank for rank in tp_rank_list], label="TP BW", linestyle=":", linewidth=3
)
plt.xlabel("Number of Devices")
plt.ylabel("Total Bandwidth (GB/s)")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.savefig("plots/comm_bw.png", bbox_inches="tight")
plt.close()

