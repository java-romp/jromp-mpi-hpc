#!/bin/bash

function check_modules() {
  local modules=("Java/21.0.2" "gradle_8.10.2")
  local module

  for module in "${modules[@]}"; do
    if ! module is-loaded "$module"; then
      echo "Module $module is not loaded. Please load it before running this script."
      exit 1
    fi
  done
}

function main() {
  local nodes=16                                     # Number of nodes
  local n_tasks=32                                   # Number of processes (MPI ranks)
  local cpus_per_task=32                             # Threads that JROMP will use
  local main_class="jromp.mpi.examples.FullParallel" # Main class to run

  local n_tasks_per_node=$((n_tasks / nodes)) # Number of processes per node
  local base_node_name="cn6"                  # Base name of the nodes
  local first_node=9                          # First node number. IMPORTANT NOTE: IceLake Rocky 8 nodes start at 9
  local node_list=()                          # List of nodes
  local node_list_str                         # List of nodes as a string separated by commas
  local i

  # Create the list of nodes
  for ((i = 0; i < nodes; i++)); do
    node_list+=("${base_node_name}$(printf "%03d" $((first_node + i)))")
  done

  # Join the elements of the array with a comma and without spaces
  node_list_str=$(
    IFS=,
    echo "${node_list[*]}"
  )

  # Set the environment variable to tell JROMP to use the number of threads
  # specified in the cpus_per_task variable
  export JROMP_NUM_THREADS=$cpus_per_task

  echo "Running with the following parameters:"
  echo "  Nodes: $nodes"
  echo "  Processes: $n_tasks"
  echo "  Threads per process: $cpus_per_task"
  echo "  Processes per node: $n_tasks_per_node"
  echo "  Node list: $node_list_str"
  echo "  Main class: $main_class"

  # Run the slurm batch
  sbatch \
    --nodes=$nodes \
    --ntasks=$n_tasks \
    --ntasks-per-node=$n_tasks_per_node \
    --cpus-per-task=$cpus_per_task \
    --nodelist="$node_list_str" \
    run.slurm $main_class
}

check_modules
main "$@"
