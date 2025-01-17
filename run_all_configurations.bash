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
#   $1: Number of nodes
#   $2: Number of tasks
#   $3: Number of cpus per task
#   $4: Node list
function run_config {
  local nodes=$1
  local n_tasks=$2
  local cpus_per_task=$3
  local node_list_str=$4
  local main_class="jromp.mpi.examples.gemm.Gemm"

  echo "Running configuration: nodes=$nodes, n_tasks=$n_tasks, cpus_per_task=$cpus_per_task, node_list=$node_list_str"

  sbatch \
    --nodes=$nodes \
    --ntasks=$n_tasks \
    --ntasks-per-node=2 \
    --cpus-per-task=$cpus_per_task \
    --nodelist="$node_list_str" \
    run.slurm $main_class $MATRIX_SIZE $cpus_per_task
}

function parallel_1 {
  run_config 1 2 1 "cn6009"
  run_config 1 2 2 "cn6010"
  run_config 1 2 4 "cn6011"
  run_config 1 2 8 "cn6012"
  run_config 1 2 12 "cn6013"
  run_config 1 2 16 "cn6014"
  run_config 1 2 24 "cn6015"
  run_config 1 2 32 "cn6016"
}

function parallel_5 {
  local nodes_9_to_11="cn[6009-6011]"
  local nodes_12_to_14="cn[6012-6014]"
  local nodes_15_to_17="cn[6015-6017]"
  local nodes_18_to_20="cn[6018-6020]"
  local nodes_21_to_23="cn[6021-6023]"

  run_config 3 6 1 "$nodes_9_to_11"
  run_config 3 6 2 "$nodes_12_to_14"
  run_config 3 6 4 "$nodes_15_to_17"
  run_config 3 6 8 "$nodes_18_to_20"
  run_config 3 6 12 "$nodes_21_to_23"
  run_config 3 6 16 "$nodes_9_to_11"
  run_config 3 6 24 "$nodes_12_to_14"
  run_config 3 6 32 "$nodes_15_to_17"
}

function parallel_10 {
  local nodes_9_to_14="cn[6009-6014]"
  local nodes_15_to_20="cn[6015-6020]"

  run_config 6 11 1 "$nodes_9_to_14"
  run_config 6 11 2 "$nodes_15_to_20"
  run_config 6 11 4 "$nodes_9_to_14"
  run_config 6 11 8 "$nodes_15_to_20"
  run_config 6 11 12 "$nodes_9_to_14"
  run_config 6 11 16 "$nodes_15_to_20"
  run_config 6 11 24 "$nodes_9_to_14"
  run_config 6 11 32 "$nodes_15_to_20"
}

function parallel_15 {
  local nodes_9_to_16="cn[6009-6016]"
  local nodes_17_to_24="cn[6017-6024]"

  run_config 8 16 1 "$nodes_9_to_16"
  run_config 8 16 2 "$nodes_17_to_24"
  run_config 8 16 4 "$nodes_9_to_16"
  run_config 8 16 8 "$nodes_17_to_24"
  run_config 8 16 12 "$nodes_9_to_16"
  run_config 8 16 16 "$nodes_17_to_24"
  run_config 8 16 24 "$nodes_9_to_16"
  run_config 8 16 32 "$nodes_17_to_24"
}

function parallel_20 {
  local nodes_9_to_20="cn[6009-6020]"

  run_config 11 21 1 "$nodes_9_to_20"
  run_config 11 21 2 "$nodes_9_to_20"
  run_config 11 21 4 "$nodes_9_to_20"
  run_config 11 21 8 "$nodes_9_to_20"
  run_config 11 21 12 "$nodes_9_to_20"
  run_config 11 21 16 "$nodes_9_to_20"
  run_config 11 21 24 "$nodes_9_to_20"
  run_config 11 21 32 "$nodes_9_to_20"
}

function parallel_25 {
  local nodes_9_to_21="cn[6009-6021]"

  run_config 13 26 1 "$nodes_9_to_21"
  run_config 13 26 2 "$nodes_9_to_21"
  run_config 13 26 4 "$nodes_9_to_21"
  run_config 13 26 8 "$nodes_9_to_21"
  run_config 13 26 12 "$nodes_9_to_21"
  run_config 13 26 16 "$nodes_9_to_21"
  run_config 13 26 24 "$nodes_9_to_21"
  run_config 13 26 32 "$nodes_9_to_21"
}

function parallel_30 {
  local all_nodes="cn[6009-6024]"

  run_config 16 31 1 "$all_nodes"
  run_config 16 31 2 "$all_nodes"
  run_config 16 31 4 "$all_nodes"
  run_config 16 31 8 "$all_nodes"
  run_config 16 31 12 "$all_nodes"
  run_config 16 31 16 "$all_nodes"
  run_config 16 31 24 "$all_nodes"
  run_config 16 31 32 "$all_nodes"
}

function main {
  parallel_1
  sleep 2 # Ensure that the previous jobs are submitted before the next ones
  parallel_5
  parallel_10
  parallel_15
  parallel_20
  parallel_25
  parallel_30
}

main "$@"

# Last revision (scastd): 16/01/2025 03:47:00
