#!/bin/bash

function check_modules() {
  local modules=("Java/21.0.2" "gradle_8.10.2" "GCC/11.2.0" "GCCcore/11.2.0")
  local module

  for module in "${modules[@]}"; do
    if ! module is-loaded "$module"; then
      echo "Module $module is not loaded. Please load it before running this script."
      exit 1
    fi
  done
}

function ceil_div() {
  local dividend=$1
  local divisor=$2
  local result=$((dividend / divisor))

  if ((dividend % divisor != 0)); then
    ((result++))
  fi

  echo $result
}

function main() {
  local nodes=16             # Number of nodes
  local n_tasks=31           # Number of processes (MPI ranks) (-1 for master)
  local cpus_per_task=32     # Threads that JROMP will use
  local n_tasks_per_node     # Number of processes per node
  local base_node_name="cn6" # Base name of the nodes
  local first_node=9         # First node number. IMPORTANT NOTE: IceLake Rocky 8 nodes start at 9
  local node_list=()         # List of nodes
  local node_list_str        # List of nodes as a string separated by commas
  local i

  local matrix_size=$((30 * 32 * 30)) # 30 tasks * 32 threads/task * number to obtain a bigger one
  local optimization_level=0
  n_tasks_per_node=$(ceil_div $n_tasks $nodes)

  # Create the list of nodes
  for ((i = 0; i < nodes; i++)); do
    node_list+=("${base_node_name}$(printf "%03d" $((first_node + i)))")
  done

  # Join the elements of the array with a comma and without spaces
  node_list_str=$(
    IFS=,
    echo "${node_list[*]}"
  )

  echo "Running with the following parameters:"
  echo "  Nodes: $nodes"
  echo "  Processes: $n_tasks ($((n_tasks - 1)) workers + 1 master)"
  echo "  Threads per process: $cpus_per_task"
  echo "  Processes per node: $n_tasks_per_node"
  echo "  Node list: $node_list_str"
  echo " ****** Program parameters ******"
  echo "  Matrix size: $matrix_size"
  echo "  Optimization level: $optimization_level"
  echo "  *******************************"

  # Run the slurm batch
  sbatch \
    --nodes=$nodes \
    --ntasks=$n_tasks \
    --ntasks-per-node="$n_tasks_per_node" \
    --cpus-per-task=$cpus_per_task \
    --nodelist="$node_list_str" \
    run.slurm $n_tasks $matrix_size $cpus_per_task $optimization_level
}

check_modules
main "$@"
