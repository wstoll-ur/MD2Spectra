#!/usr/bin/env bash
#SBATCH --job-name=md2spectra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
# Adapt account, partition, modules and environment to your cluster.
# Install MD2Spectra and authenticate to Hugging Face before submitting.
# Usage: sbatch examples/slurm.sh structure.cif runs/structure-300K
: "${1:?Provide a CIF path}"
: "${2:?Provide an output directory}"
mkdir -p "$2"
# A scheduler warning creates STOP; the MD subprocess checks it each saved frame.
# Remove an old STOP manually before resubmitting a deliberately stopped run.
trap 'touch "$2/STOP"' TERM INT
md2spectra run --cif "$1" --temperature 300 --supercell 2 1 2 \
  --equilibration-ps 10 --production-ps 100 --segment-ps 20 \
  --device cuda --output "$2" &
child=$!
# wait can be interrupted by the warning signal; allow the child to checkpoint.
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
  wait "$child"
  status=$?
fi
exit "$status"
