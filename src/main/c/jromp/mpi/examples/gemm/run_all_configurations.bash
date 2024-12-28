#!/bin/bash

MATRIX_SIZE=$((30 * 32 * 30)) # 30 tasks * 32 threads/task * number to obtain a bigger one

function ceil_div {
  local dividend=$1
  local divisor=$2
  local result=$((dividend / divisor))

  if ((dividend % divisor != 0)); then
    ((result++))
  fi

  echo $result
}

# Execute the slurm batch
# Parameters:
#   $1: Number of nodes (1 8 16)
#   $2: Number of tasks (2 16 31)
#   $3: Number of cpus per task (1 12 32)
#   $4: Optimization level (0 1 2 3)
#   $5: Node list
function run_config {
  local nodes=$1
  local n_tasks=$2
  local cpus_per_task=$3
  local optimization_level=$4
  local node_list_str=$5

  echo "Running configuration: nodes=$nodes, n_tasks=$n_tasks, cpus_per_task=$cpus_per_task, optimization_level=$optimization_level, node_list=$node_list_str"

  local compiled_file="gemm_${nodes}_${n_tasks}_${MATRIX_SIZE}_${cpus_per_task}_${optimization_level}"

  sbatch \
    --nodes=$nodes \
    --ntasks=$n_tasks \
    --ntasks-per-node=2 \
    --cpus-per-task=$cpus_per_task \
    --nodelist="$node_list_str" \
    run.slurm $n_tasks $MATRIX_SIZE $cpus_per_task $optimization_level $compiled_file
}

function sequential {
  run_config 1 2 1 0 "cn6009"
  run_config 1 2 1 1 "cn6010"
  run_config 1 2 1 2 "cn6011"
  run_config 1 2 1 3 "cn6012"
  run_config 1 2 12 0 "cn6013"
  run_config 1 2 12 1 "cn6014"
  run_config 1 2 12 2 "cn6015"
  run_config 1 2 12 3 "cn6016"
  run_config 1 2 32 0 "cn6017"
  run_config 1 2 32 1 "cn6018"
  run_config 1 2 32 2 "cn6019"
  run_config 1 2 32 3 "cn6020"
}

function sequential_1 {
  run_config 1 2 4 0 "cn6009"
  run_config 1 2 4 1 "cn6010"
  run_config 1 2 4 2 "cn6011"
  run_config 1 2 4 3 "cn6012"
  run_config 1 2 8 0 "cn6013"
  run_config 1 2 8 1 "cn6014"
  run_config 1 2 8 2 "cn6015"
  run_config 1 2 8 3 "cn6016"
  run_config 1 2 16 0 "cn6017"
  run_config 1 2 16 1 "cn6018"
  run_config 1 2 16 2 "cn6019"
  run_config 1 2 16 3 "cn6020"
}

function parallel_1 {
  local nodes_9_to_16="cn[6009-6016]"
  local nodes_17_to_24="cn[6017-6024]"

  run_config 8 16 1 0 "$nodes_9_to_16"
  run_config 8 16 1 1 "$nodes_17_to_24"
  run_config 8 16 1 2 "$nodes_9_to_16"
  run_config 8 16 1 3 "$nodes_17_to_24"
  run_config 8 16 12 0 "$nodes_9_to_16"
  run_config 8 16 12 1 "$nodes_17_to_24"
  run_config 8 16 12 2 "$nodes_9_to_16"
  run_config 8 16 12 3 "$nodes_17_to_24"
  run_config 8 16 32 0 "$nodes_9_to_16"
  run_config 8 16 32 1 "$nodes_17_to_24"
  run_config 8 16 32 2 "$nodes_9_to_16"
  run_config 8 16 32 3 "$nodes_17_to_24"
}

function parallel_1_1 {
  local nodes_9_to_16="cn[6009-6016]"
  local nodes_17_to_24="cn[6017-6024]"

  run_config 8 16 4 0 "$nodes_9_to_16"
  run_config 8 16 4 1 "$nodes_17_to_24"
  run_config 8 16 4 2 "$nodes_9_to_16"
  run_config 8 16 4 3 "$nodes_17_to_24"
  run_config 8 16 8 0 "$nodes_9_to_16"
  run_config 8 16 8 1 "$nodes_17_to_24"
  run_config 8 16 8 2 "$nodes_9_to_16"
  run_config 8 16 8 3 "$nodes_17_to_24"
  run_config 8 16 16 0 "$nodes_9_to_16"
  run_config 8 16 16 1 "$nodes_17_to_24"
  run_config 8 16 16 2 "$nodes_9_to_16"
  run_config 8 16 16 3 "$nodes_17_to_24"
}

function parallel_2 {
  local all_nodes="cn[6009-6024]"

  run_config 16 31 1 0 "$all_nodes"
  run_config 16 31 1 1 "$all_nodes"
  run_config 16 31 1 2 "$all_nodes"
  run_config 16 31 1 3 "$all_nodes"
  run_config 16 31 12 0 "$all_nodes"
  run_config 16 31 12 1 "$all_nodes"
  run_config 16 31 12 2 "$all_nodes"
  run_config 16 31 12 3 "$all_nodes"
  run_config 16 31 32 0 "$all_nodes"
  run_config 16 31 32 1 "$all_nodes"
  run_config 16 31 32 2 "$all_nodes"
  run_config 16 31 32 3 "$all_nodes"
}

function parallel_2_1 {
  local all_nodes="cn[6009-6024]"

  run_config 16 31 4 0 "$all_nodes"
  run_config 16 31 4 1 "$all_nodes"
  run_config 16 31 4 2 "$all_nodes"
  run_config 16 31 4 3 "$all_nodes"
  run_config 16 31 8 0 "$all_nodes"
  run_config 16 31 8 1 "$all_nodes"
  run_config 16 31 8 2 "$all_nodes"
  run_config 16 31 8 3 "$all_nodes"
  run_config 16 31 16 0 "$all_nodes"
  run_config 16 31 16 1 "$all_nodes"
  run_config 16 31 16 2 "$all_nodes"
  run_config 16 31 16 3 "$all_nodes"
}

function main {
  sequential_1
  sleep 3 # Warranty that the previous jobs are already running
  parallel_1_1
  sleep 3
  parallel_2_1
}

main "$@"
