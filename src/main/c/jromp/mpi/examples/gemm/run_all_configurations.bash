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

function parallel_1 {
  run_config 1 2 1 0 "cn6009"
  run_config 1 2 1 1 "cn6010"
  run_config 1 2 1 2 "cn6011"
  run_config 1 2 1 3 "cn6012"

  run_config 1 2 2 0 "cn6013"
  run_config 1 2 2 1 "cn6014"
  run_config 1 2 2 2 "cn6015"
  run_config 1 2 2 3 "cn6016"

  run_config 1 2 4 0 "cn6017"
  run_config 1 2 4 1 "cn6018"
  run_config 1 2 4 2 "cn6019"
  run_config 1 2 4 3 "cn6020"

  run_config 1 2 8 0 "cn6021"
  run_config 1 2 8 1 "cn6022"
  run_config 1 2 8 2 "cn6023"
  run_config 1 2 8 3 "cn6024"

  run_config 1 2 12 0 "cn6009"
  run_config 1 2 12 1 "cn6010"
  run_config 1 2 12 2 "cn6011"
  run_config 1 2 12 3 "cn6012"

  run_config 1 2 16 0 "cn6013"
  run_config 1 2 16 1 "cn6014"
  run_config 1 2 16 2 "cn6015"
  run_config 1 2 16 3 "cn6016"

  run_config 1 2 20 0 "cn6017"
  run_config 1 2 20 1 "cn6018"
  run_config 1 2 20 2 "cn6019"
  run_config 1 2 20 3 "cn6020"

  run_config 1 2 24 0 "cn6021"
  run_config 1 2 24 1 "cn6022"
  run_config 1 2 24 2 "cn6023"
  run_config 1 2 24 3 "cn6024"

  run_config 1 2 32 0 "cn6009"
  run_config 1 2 32 1 "cn6010"
  run_config 1 2 32 2 "cn6011"
  run_config 1 2 32 3 "cn6012"
}

function parallel_5 {
  local nodes_9_to_11="cn[6009-6011]"
  local nodes_12_to_14="cn[6012-6014]"
  local nodes_15_to_17="cn[6015-6017]"
  local nodes_18_to_20="cn[6018-6020]"
  local nodes_21_to_23="cn[6021-6023]"

  run_config 3 6 1 0 "$nodes_9_to_11"
  run_config 3 6 1 1 "$nodes_12_to_14"
  run_config 3 6 1 2 "$nodes_15_to_17"
  run_config 3 6 1 3 "$nodes_18_to_20"

  run_config 3 6 2 0 "$nodes_21_to_23"
  run_config 3 6 2 1 "$nodes_9_to_11"
  run_config 3 6 2 2 "$nodes_12_to_14"
  run_config 3 6 2 3 "$nodes_15_to_17"

  run_config 3 6 4 0 "$nodes_18_to_20"
  run_config 3 6 4 1 "$nodes_21_to_23"
  run_config 3 6 4 2 "$nodes_9_to_11"
  run_config 3 6 4 3 "$nodes_12_to_14"

  run_config 3 6 8 0 "$nodes_15_to_17"
  run_config 3 6 8 1 "$nodes_18_to_20"
  run_config 3 6 8 2 "$nodes_21_to_23"
  run_config 3 6 8 3 "$nodes_9_to_11"

  run_config 3 6 12 0 "$nodes_12_to_14"
  run_config 3 6 12 1 "$nodes_15_to_17"
  run_config 3 6 12 2 "$nodes_18_to_20"
  run_config 3 6 12 3 "$nodes_21_to_23"

  run_config 3 6 16 0 "$nodes_9_to_11"
  run_config 3 6 16 1 "$nodes_12_to_14"
  run_config 3 6 16 2 "$nodes_15_to_17"
  run_config 3 6 16 3 "$nodes_18_to_20"

  run_config 3 6 20 0 "$nodes_21_to_23"
  run_config 3 6 20 1 "$nodes_9_to_11"
  run_config 3 6 20 2 "$nodes_12_to_14"
  run_config 3 6 20 3 "$nodes_15_to_17"

  run_config 3 6 24 0 "$nodes_18_to_20"
  run_config 3 6 24 1 "$nodes_21_to_23"
  run_config 3 6 24 2 "$nodes_15_to_17"
  run_config 3 6 24 3 "$nodes_9_to_11"

  run_config 3 6 32 0 "$nodes_12_to_14"
  run_config 3 6 32 1 "$nodes_15_to_17"
  run_config 3 6 32 2 "$nodes_18_to_20"
  run_config 3 6 32 3 "$nodes_9_to_11"
}

function parallel_10 {
  local nodes_9_to_14="cn[6009-6014]"
  local nodes_15_to_20="cn[6015-6020]"

  run_config 6 11 1 0 "$nodes_9_to_14"
  run_config 6 11 1 1 "$nodes_15_to_20"
  run_config 6 11 1 2 "$nodes_9_to_14"
  run_config 6 11 1 3 "$nodes_15_to_20"

  run_config 6 11 2 0 "$nodes_9_to_14"
  run_config 6 11 2 1 "$nodes_15_to_20"
  run_config 6 11 2 2 "$nodes_9_to_14"
  run_config 6 11 2 3 "$nodes_15_to_20"

  run_config 6 11 4 0 "$nodes_9_to_14"
  run_config 6 11 4 1 "$nodes_15_to_20"
  run_config 6 11 4 2 "$nodes_9_to_14"
  run_config 6 11 4 3 "$nodes_15_to_20"

  run_config 6 11 8 0 "$nodes_9_to_14"
  run_config 6 11 8 1 "$nodes_15_to_20"
  run_config 6 11 8 2 "$nodes_9_to_14"
  run_config 6 11 8 3 "$nodes_15_to_20"

  run_config 6 11 12 0 "$nodes_9_to_14"
  run_config 6 11 12 1 "$nodes_15_to_20"
  run_config 6 11 12 2 "$nodes_9_to_14"
  run_config 6 11 12 3 "$nodes_15_to_20"

  run_config 6 11 16 0 "$nodes_9_to_14"
  run_config 6 11 16 1 "$nodes_15_to_20"
  run_config 6 11 16 2 "$nodes_9_to_14"
  run_config 6 11 16 3 "$nodes_15_to_20"

  run_config 6 11 20 0 "$nodes_9_to_14"
  run_config 6 11 20 1 "$nodes_15_to_20"
  run_config 6 11 20 2 "$nodes_9_to_14"
  run_config 6 11 20 3 "$nodes_15_to_20"

  run_config 6 11 24 0 "$nodes_9_to_14"
  run_config 6 11 24 1 "$nodes_15_to_20"
  run_config 6 11 24 2 "$nodes_9_to_14"
  run_config 6 11 24 3 "$nodes_15_to_20"

  run_config 6 11 32 0 "$nodes_9_to_14"
  run_config 6 11 32 1 "$nodes_15_to_20"
  run_config 6 11 32 2 "$nodes_9_to_14"
  run_config 6 11 32 3 "$nodes_15_to_20"
}

function parallel_15 {
  local nodes_9_to_16="cn[6009-6016]"
  local nodes_17_to_24="cn[6017-6024]"

  run_config 8 16 1 0 "$nodes_9_to_16"
  run_config 8 16 1 1 "$nodes_17_to_24"
  run_config 8 16 1 2 "$nodes_9_to_16"
  run_config 8 16 1 3 "$nodes_17_to_24"

  run_config 8 16 2 0 "$nodes_9_to_16"
  run_config 8 16 2 1 "$nodes_17_to_24"
  run_config 8 16 2 2 "$nodes_9_to_16"
  run_config 8 16 2 3 "$nodes_17_to_24"

  run_config 8 16 4 0 "$nodes_9_to_16"
  run_config 8 16 4 1 "$nodes_17_to_24"
  run_config 8 16 4 2 "$nodes_9_to_16"
  run_config 8 16 4 3 "$nodes_17_to_24"

  run_config 8 16 8 0 "$nodes_9_to_16"
  run_config 8 16 8 1 "$nodes_17_to_24"
  run_config 8 16 8 2 "$nodes_9_to_16"
  run_config 8 16 8 3 "$nodes_17_to_24"

  run_config 8 16 12 0 "$nodes_9_to_16"
  run_config 8 16 12 1 "$nodes_17_to_24"
  run_config 8 16 12 2 "$nodes_9_to_16"
  run_config 8 16 12 3 "$nodes_17_to_24"

  run_config 8 16 16 0 "$nodes_9_to_16"
  run_config 8 16 16 1 "$nodes_17_to_24"
  run_config 8 16 16 2 "$nodes_9_to_16"
  run_config 8 16 16 3 "$nodes_17_to_24"

  run_config 8 16 20 0 "$nodes_9_to_16"
  run_config 8 16 20 1 "$nodes_17_to_24"
  run_config 8 16 20 2 "$nodes_9_to_16"
  run_config 8 16 20 3 "$nodes_17_to_24"

  run_config 8 16 24 0 "$nodes_9_to_16"
  run_config 8 16 24 1 "$nodes_17_to_24"
  run_config 8 16 24 2 "$nodes_9_to_16"
  run_config 8 16 24 3 "$nodes_17_to_24"

  run_config 8 16 32 0 "$nodes_9_to_16"
  run_config 8 16 32 1 "$nodes_17_to_24"
  run_config 8 16 32 2 "$nodes_9_to_16"
  run_config 8 16 32 3 "$nodes_17_to_24"
}

function parallel_20 {
  local nodes_9_to_20="cn[6009-6020]"

  run_config 11 21 1 0 "$nodes_9_to_20"
  run_config 11 21 1 1 "$nodes_9_to_20"
  run_config 11 21 1 2 "$nodes_9_to_20"
  run_config 11 21 1 3 "$nodes_9_to_20"

  run_config 11 21 2 0 "$nodes_9_to_20"
  run_config 11 21 2 1 "$nodes_9_to_20"
  run_config 11 21 2 2 "$nodes_9_to_20"
  run_config 11 21 2 3 "$nodes_9_to_20"

  run_config 11 21 4 0 "$nodes_9_to_20"
  run_config 11 21 4 1 "$nodes_9_to_20"
  run_config 11 21 4 2 "$nodes_9_to_20"
  run_config 11 21 4 3 "$nodes_9_to_20"

  run_config 11 21 8 0 "$nodes_9_to_20"
  run_config 11 21 8 1 "$nodes_9_to_20"
  run_config 11 21 8 2 "$nodes_9_to_20"
  run_config 11 21 8 3 "$nodes_9_to_20"

  run_config 11 21 12 0 "$nodes_9_to_20"
  run_config 11 21 12 1 "$nodes_9_to_20"
  run_config 11 21 12 2 "$nodes_9_to_20"
  run_config 11 21 12 3 "$nodes_9_to_20"

  run_config 11 21 16 0 "$nodes_9_to_20"
  run_config 11 21 16 1 "$nodes_9_to_20"
  run_config 11 21 16 2 "$nodes_9_to_20"
  run_config 11 21 16 3 "$nodes_9_to_20"

  run_config 11 21 20 0 "$nodes_9_to_20"
  run_config 11 21 20 1 "$nodes_9_to_20"
  run_config 11 21 20 2 "$nodes_9_to_20"
  run_config 11 21 20 3 "$nodes_9_to_20"

  run_config 11 21 24 0 "$nodes_9_to_20"
  run_config 11 21 24 1 "$nodes_9_to_20"
  run_config 11 21 24 2 "$nodes_9_to_20"
  run_config 11 21 24 3 "$nodes_9_to_20"

  run_config 11 21 32 0 "$nodes_9_to_20"
  run_config 11 21 32 1 "$nodes_9_to_20"
  run_config 11 21 32 2 "$nodes_9_to_20"
  run_config 11 21 32 3 "$nodes_9_to_20"
}

function parallel_25 {
  local nodes_9_to_21="cn[6009-6021]"

  run_config 13 26 1 0 "$nodes_9_to_21"
  run_config 13 26 1 1 "$nodes_9_to_21"
  run_config 13 26 1 2 "$nodes_9_to_21"
  run_config 13 26 1 3 "$nodes_9_to_21"

  run_config 13 26 2 0 "$nodes_9_to_21"
  run_config 13 26 2 1 "$nodes_9_to_21"
  run_config 13 26 2 2 "$nodes_9_to_21"
  run_config 13 26 2 3 "$nodes_9_to_21"

  run_config 13 26 4 0 "$nodes_9_to_21"
  run_config 13 26 4 1 "$nodes_9_to_21"
  run_config 13 26 4 2 "$nodes_9_to_21"
  run_config 13 26 4 3 "$nodes_9_to_21"

  run_config 13 26 8 0 "$nodes_9_to_21"
  run_config 13 26 8 1 "$nodes_9_to_21"
  run_config 13 26 8 2 "$nodes_9_to_21"
  run_config 13 26 8 3 "$nodes_9_to_21"

  run_config 13 26 12 0 "$nodes_9_to_21"
  run_config 13 26 12 1 "$nodes_9_to_21"
  run_config 13 26 12 2 "$nodes_9_to_21"
  run_config 13 26 12 3 "$nodes_9_to_21"

  run_config 13 26 16 0 "$nodes_9_to_21"
  run_config 13 26 16 1 "$nodes_9_to_21"
  run_config 13 26 16 2 "$nodes_9_to_21"
  run_config 13 26 16 3 "$nodes_9_to_21"

  # 20 threads cannot divide

  run_config 13 26 24 0 "$nodes_9_to_21"
  run_config 13 26 24 1 "$nodes_9_to_21"
  run_config 13 26 24 2 "$nodes_9_to_21"
  run_config 13 26 24 3 "$nodes_9_to_21"

  run_config 13 26 32 0 "$nodes_9_to_21"
  run_config 13 26 32 1 "$nodes_9_to_21"
  run_config 13 26 32 2 "$nodes_9_to_21"
  run_config 13 26 32 3 "$nodes_9_to_21"
}

function parallel_30 {
  local all_nodes="cn[6009-6024]"

  run_config 16 31 1 0 "$all_nodes"
  run_config 16 31 1 1 "$all_nodes"
  run_config 16 31 1 2 "$all_nodes"
  run_config 16 31 1 3 "$all_nodes"

  run_config 16 31 2 0 "$all_nodes"
  run_config 16 31 2 1 "$all_nodes"
  run_config 16 31 2 2 "$all_nodes"
  run_config 16 31 2 3 "$all_nodes"

  run_config 16 31 4 0 "$all_nodes"
  run_config 16 31 4 1 "$all_nodes"
  run_config 16 31 4 2 "$all_nodes"
  run_config 16 31 4 3 "$all_nodes"

  run_config 16 31 8 0 "$all_nodes"
  run_config 16 31 8 1 "$all_nodes"
  run_config 16 31 8 2 "$all_nodes"
  run_config 16 31 8 3 "$all_nodes"

  run_config 16 31 12 0 "$all_nodes"
  run_config 16 31 12 1 "$all_nodes"
  run_config 16 31 12 2 "$all_nodes"
  run_config 16 31 12 3 "$all_nodes"

  run_config 16 31 16 0 "$all_nodes"
  run_config 16 31 16 1 "$all_nodes"
  run_config 16 31 16 2 "$all_nodes"
  run_config 16 31 16 3 "$all_nodes"

  run_config 16 31 20 0 "$all_nodes"
  run_config 16 31 20 1 "$all_nodes"
  run_config 16 31 20 2 "$all_nodes"
  run_config 16 31 20 3 "$all_nodes"

  run_config 16 31 24 0 "$all_nodes"
  run_config 16 31 24 1 "$all_nodes"
  run_config 16 31 24 2 "$all_nodes"
  run_config 16 31 24 3 "$all_nodes"

  run_config 16 31 32 0 "$all_nodes"
  run_config 16 31 32 1 "$all_nodes"
  run_config 16 31 32 2 "$all_nodes"
  run_config 16 31 32 3 "$all_nodes"
}

function main {
  parallel_1
  sleep 2
  parallel_5
  sleep 2
  parallel_10
  sleep 2
  parallel_15
  sleep 2
  parallel_20
  sleep 2
  parallel_25
  sleep 2
  parallel_30
}

main "$@"
