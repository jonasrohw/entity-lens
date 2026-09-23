#!/bin/bash
# Pull jobs off a queue file with ONE WORKER PINNED PER GPU.
#
#   analysis/run_pinned.sh JOBFILE N_GPUS
#
# Use this only on a machine whose cards are all ours.  run_queue.sh claims
# a free card per job, which is right on a shared machine but races when we
# own every card: free_gpu reads nvidia-smi, and a card claimed 30 seconds
# ago still looks idle while its model loads, so two workers take the same
# card and both die of OOM.  That is exactly what happened on rho1 --
# gpu2 was handed out five times and three relations were lost.
#
# Pinning removes the race by construction: worker i owns GPU i for its
# whole life and never consults nvidia-smi.  Jobs are handed out under a
# lock on a shared cursor, so each runs exactly once.
cd "$(dirname "$(readlink -f "$0")")/../.."
Q=$1; NG=${2:-8}
LOCK="$Q.lock"; CUR="$Q.cursor"
: >| "$LOCK"; echo 1 >| "$CUR"

worker() {
  local g=$1 n line
  while true; do
    n=$( flock "$LOCK" bash -c "n=\$(cat $CUR); echo \$((n+1)) >| $CUR; echo \$n" )
    line=$(sed -n "${n}p" "$Q"); [ -z "$line" ] && break
    echo "[$(date '+%H:%M')] gpu$g job $n: $(echo "$line" | grep -o '\--relation [a-z_]*' | head -1)"
    CUDA_VISIBLE_DEVICES=$g bash -c "$line"
    echo "[$(date '+%H:%M')] gpu$g job $n done rc=$?"
  done
  echo "[$(date '+%H:%M')] gpu$g finished"
}
for g in $(seq 0 $((NG-1))); do worker "$g" & done
wait
echo "[$(date '+%H:%M')] ALL WORKERS FINISHED"
