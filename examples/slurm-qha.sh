#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Submit: bash examples/slurm-qha.sh submit crystal.cif /absolute/run-directory
# UMA_PYTHON and RESPONSE_PYTHON select installed environments. Set UMA_MODEL,
# MDP_MODEL, ACCOUNT and PARTITION if required by the cluster. Each array task
# uses one GPU. Load CUDA/modules before submission as required by your cluster.
set -euo pipefail
phase=${1:?Expected submit, qha or temperature}
cif=${2:?Expected CIF path}
output=${3:?Expected output directory}
export UMA_PYTHON=${UMA_PYTHON:-python}
export RESPONSE_PYTHON=${RESPONSE_PYTHON:-$UMA_PYTHON}
export UMA_MODEL=${UMA_MODEL:-uma-s-1p2p1}
export MDP_MODEL=${MDP_MODEL:-auto}
case "$phase" in
 submit)
  script=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
  cif=$("$UMA_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$cif")
  mkdir -p -- "$output/logs"
  output=$(cd -- "$output" && pwd)
  options=(--export=ALL)
  if [[ -n ${ACCOUNT:-} ]]; then options+=(--account="$ACCOUNT"); fi
  if [[ -n ${PARTITION:-} ]]; then options+=(--partition="$PARTITION"); fi
  head_job=$(sbatch --parsable "${options[@]}" --job-name=md2-gamma-qha \
    --output="$output/logs/qha-%j.out" "$script" qha "$cif" "$output")
  head_job=${head_job%%;*}
  temperatures_job=$(sbatch --parsable "${options[@]}" --job-name=md2-temperature \
    --array=0-5 --dependency="afterok:$head_job" \
    --output="$output/logs/temperature-%A_%a.out" "$script" temperature "$cif" "$output")
  printf 'Gamma QHA job: %s\nTemperature array: %s\n' "$head_job" "$temperatures_job"
  ;;
 qha)
  exec "$UMA_PYTHON" -m md2spectra qha --cif "$cif" --output "$output/qha" \
    --uma-model "$UMA_MODEL" --device cuda --task omc \
    --qha-relax-reference --qha-phonon-sampling gamma --qha-supercell 1 1 1 \
    --qha-volume-factors 0.98 0.99 1.00 1.01 1.02 1.03 1.04 \
    --qha-fmax 0.0001 --qha-shape-stress-gpa 0.01 \
    --temperatures 78 100 150 200 250 298
  ;;
 temperature)
  exec "$UMA_PYTHON" - "$output" "${SLURM_ARRAY_TASK_ID:?Expected array index}" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from md2spectra.io import sha256
root = Path(sys.argv[1])
result = json.loads((root/'qha/qha-complete.json').read_text())
row = result['temperatures'][int(sys.argv[2])]
structure = root/'qha'/row['structure']
if sha256(structure) != row['structure_sha256']:
    raise ValueError('Temperature-cell checksum changed')
subprocess.run([sys.executable, '-m', 'md2spectra', 'run',
    '--cif', str(structure), '--output', str(root/f"temperature-{row['temperature_K']:g}K"),
    '--temperature', str(row['temperature_K']), '--supercell', '2', '2', '2',
    '--no-relax', '--uma-model', os.environ['UMA_MODEL'], '--uma-inference', 'md',
    '--mdp-model', os.environ['MDP_MODEL'], '--response-python', os.environ['RESPONSE_PYTHON'],
    '--device', 'cuda', '--raman-only', '--laser-nm', '785',
    '--equilibration-ps', '10', '--production-ps', '200', '--timestep-fs', '0.5',
    '--save-every', '4', '--no-save-momenta', '--segment-ps', '20'], check=True)
PY
  ;;
 *) printf 'Unknown stage: %s\n' "$phase" >&2; exit 2 ;;
esac
