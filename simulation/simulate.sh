DEVICE=(16 32 64 128 256 512 1024)
MODEL=("facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b")
BWIN=(5 10 20 50 100 200)
BWOUT=(1 2 5 10)
FLOPS=(1 3 6 9 12 15)
BS=(16 32 64 128 256 512)
SEQLEN=(128 256 512 1024 2048)

# truncate the output file results.csv
echo "Model,Devices,TFLOPS,BWin,BWout,BS,SeqLen,Method,Latency" > results.csv

for model in "${MODEL[@]}"; do
  for device in "${DEVICE[@]}"; do
    for flops in "${FLOPS[@]}"; do
      for bwin in "${BWIN[@]}"; do
        for bwout in "${BWOUT[@]}"; do
          for bs in "${BS[@]}"; do
            for seq in "${SEQLEN[@]}"; do
                echo "Running $model on $device devices with $flops TFLOPS, $bwin BWin, $bwout BWout, $bs BS, $seq SeqLen"
                python simulate.py --model $model --num_devices $device --flops $flops --in_bw $bwin --out_bw $bwout --batch_size $bs --dtype="half" --seq_len $seq > /tmp/tmp_${bs}_${seq}.txt &
            done
          done
          # wait for all the simulations to finish'
          wait
          for bs in "${BS[@]}"; do
            for seq in "${SEQLEN[@]}"; do
              echo "Parsing $model on $device devices with $flops TFLOPS, $bwin BWin, $bwout BWout, $bs BS, $seq SeqLen"
              latency="$(python result_parser.py /tmp/tmp_${bs}_${seq}.txt)"
              hybrid_latency="$(echo $latency | cut -d' ' -f1)"
              tp_latency="$(echo $latency | cut -d' ' -f2)"
              tp_cache_latency="$(echo $latency | cut -d' ' -f3)"
              echo "$model,$device,$flops,$bwin,$bwout,$bs,$seq,Hybrid,$hybrid_latency" >> results.csv
              echo "$model,$device,$flops,$bwin,$bwout,$bs,$seq,TP,$tp_latency" >> results.csv
              echo "$model,$device,$flops,$bwin,$bwout,$bs,$seq,TPCache,$tp_cache_latency" >> results.csv
            done
          done
        done
      done
    done
  done
done
