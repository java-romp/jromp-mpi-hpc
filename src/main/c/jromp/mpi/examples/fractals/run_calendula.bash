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
  local num=$1
  local div=$2
  echo $(((num + div - 1) / div))
}

function main() {
  local nodes=16         # Number of nodes
  local n_tasks=31       # Number of processes (MPI ranks)
  local cpus_per_task=30 # Threads that JROMP will use

  local n_tasks_per_node     # Number of processes per node
  local base_node_name="cn6" # Base name of the nodes
  local first_node=9         # First node number. IMPORTANT NOTE: IceLake Rocky 8 nodes start at 9
  local node_list=()         # List of nodes
  local node_list_str        # List of nodes as a string separated by commas
  local i

  local iterations=6000
  local real_part=3.5
  local width=3000
  local height=3000
  local block_size=$((height / (n_tasks - 1)))

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
  echo "  Processes: $n_tasks"
  echo "  Threads per process: $cpus_per_task"
  echo "  Processes per node: $n_tasks_per_node"
  echo "  Node list: $node_list_str"
  echo " **** Parameters for the mandelbrot set ****"
  echo "  Iterations: $iterations"
  echo "  Real part: $real_part"
  echo "  Image size: $width x $height"
  echo "  Block size: $block_size"

  # Run the slurm batch
  sbatch \
    --nodes=$nodes \
    --ntasks=$n_tasks \
    --ntasks-per-node="$n_tasks_per_node" \
    --cpus-per-task=$cpus_per_task \
    --nodelist="$node_list_str" \
    run.slurm $iterations $real_part $width $height $block_size $n_tasks
}

check_modules
main "$@"
