import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

matplotlib.rcParams.update({'font.size': 24})

df = pd.read_csv("results.csv")
print(df)

model = "facebook/opt-30b"

df_model = df[df["Model"] == model]

model = model.split("/")[-1]

print(df_model)

# plot latency scaling with device
df_plot = df_model[(df_model["TFLOPS"] == 9) & (df_model["BS"] == 128) & (df_model["SeqLen"] == 1024) & (df_model["BWin"] == 100) & (df_model["BWout"] == 10)]
plt.figure(dpi=300, figsize=(8, 8)) 
sns.lineplot(data=df_plot, x="Devices", y="Latency", hue="Method", linewidth=3)
plt.yscale("log")
plt.xlabel("Number of Devices")
plt.ylabel("Latency (s)")
plt.title(f"Latency Scaling with Devices for {model}")
plt.savefig(f"latency_scaling_device_{model}.pdf", bbox_inches="tight")
plt.savefig(f"latency_scaling_device_{model}.png", bbox_inches="tight")
plt.close()

# plot latency scaling with batch size
df_plot = df_model[(df_model["TFLOPS"] == 9) & (df_model["Devices"] == 64) & (df_model["SeqLen"] == 1024) & (df_model["BWin"] == 100) & (df_model["BWout"] == 10)]
plt.figure(dpi=300, figsize=(8, 8)) 
sns.lineplot(data=df_plot, x="BS", y="Latency", hue="Method", linewidth=3)
plt.yscale("log")
plt.xlabel("Batch Size")
plt.ylabel("Latency (s)")
plt.title(f"Latency Scaling with Batch Size for {model}")
plt.savefig(f"latency_scaling_bs_{model}.pdf", bbox_inches="tight")
plt.savefig(f"latency_scaling_bs_{model}.png", bbox_inches="tight")
plt.close()

# plot latency scaling with BWout
df_plot = df_model[(df_model["TFLOPS"] == 9) & (df_model["Devices"] == 64) & (df_model["SeqLen"] == 1024) & (df_model["BWin"] == 100) & (df_model["BS"] == 64)]
plt.figure(dpi=300, figsize=(8, 8)) 
sns.lineplot(data=df_plot, x="BWout", y="Latency", hue="Method", linewidth=3)
plt.yscale("log")
plt.xlabel("Uplink Bandwidth (MB/s)")
plt.ylabel("Latency (s)")
plt.title(f"Latency Scaling with Bandwidth for {model}")
plt.savefig(f"latency_scaling_bwout_{model}.pdf", bbox_inches="tight")
plt.savefig(f"latency_scaling_bwout_{model}.png", bbox_inches="tight")
plt.close()

# plot latency scaling with BWin
df_plot = df_model[(df_model["TFLOPS"] == 9) & (df_model["Devices"] == 64) & (df_model["SeqLen"] == 1024) & (df_model["BWout"] == 10) & (df_model["BS"] == 64)]
plt.figure(dpi=300, figsize=(8, 8)) 
sns.lineplot(data=df_plot, x="BWin", y="Latency", hue="Method", linewidth=3)
plt.yscale("log")
plt.xlabel("Downlink Bandwidth (MB/s)")
plt.ylabel("Latency (s)")
plt.title(f"Latency Scaling with Bandwidth for {model}")
plt.savefig(f"latency_scaling_bwin_{model}.pdf", bbox_inches="tight")
plt.savefig(f"latency_scaling_bwin_{model}.png", bbox_inches="tight")
plt.close()

