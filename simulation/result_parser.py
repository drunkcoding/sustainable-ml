import re
import sys


filename = sys.argv[1]

with open(filename, 'r') as file:
    data = file.readlines()

for line in data:
    line = line.strip()
    if "Hybrid" in line:
        h_latency = float(line.split(" ")[-1])

    if "w/o" in line:
        e_latency = float(line.split(" ")[-1])

    if "w/" in line:
        c_latency = float(line.split(" ")[-1])

print(h_latency, e_latency, c_latency)