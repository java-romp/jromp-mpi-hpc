#!/bin/bash

function check_modules {
  local modules=("Java/21.0.2" "gradle_8.10.2" "GCC/11.2.0" "GCCcore/11.2.0")
  local module

  for module in "${modules[@]}"; do
    if ! module is-loaded "$module"; then
      echo "Module $module is not loaded. Please load it before running this script."
      exit 1
    fi
  done
}

function ceil_div {
  local dividend=$1
  local divisor=$2
  local result=$((dividend / divisor))

  if ((dividend % divisor != 0)); then
    ((result++))
  fi

  echo $result
}

function main {
  local nodes=(1)
  local n_tasks=(2)
  local cpus_per_task=(1 12 32)
  local optimization_level=(0 1 2 3)
  local base_node_name="cn6" # Base name of the nodes
  local first_node=24        # First node number. IMPORTANT NOTE: IceLake Rocky 8 nodes start at 9
  local node_list=()         # List of nodes
  local node_list_str        # List of nodes as a string separated by commas
  local n_tasks_per_node
  local current_node

  local matrix_size=$((30 * 32 * 30)) # 30 tasks * 32 threads/task * number to obtain a bigger one
  local n
  local t
  local c
  local o

  for n in "${nodes[@]}"; do
    for t in "${n_tasks[@]}"; do
      for c in "${cpus_per_task[@]}"; do
        for o in "${optimization_level[@]}"; do
          n_tasks_per_node=$(ceil_div $t $n)

          node_list+=("${base_node_name}$(printf "%03d" $((first_node - current_node)))")
          current_node=$((current_node + 1))

          node_list_str=$(
            IFS=,
            echo "${node_list[*]}"
          )

          echo "Running with the following parameters:"
          echo "  Nodes: $n ($node_list_str)"
          echo "  Total processes: $t ($((t - 1)) workers + 1 master)"
          echo "  Threads per process: $c"
          echo "  Processes per node: $n_tasks_per_node"
          echo " ****** Program parameters ******"
          echo "  Matrix size: $matrix_size"
          echo "  Optimization level: $o"
          echo "  *******************************"
          echo

          local compiled_file="gemm_${n}_${t}_${matrix_size}_${c}_${o}"

          # Run the slurm batch
          sbatch \
            --nodes=$n \
            --ntasks=$t \
            --ntasks-per-node="$n_tasks_per_node" \
            --cpus-per-task=$c \
            --nodelist="$node_list_str" \
            run.slurm $t $matrix_size $c $o $compiled_file

          node_list=()
        done
      done
    done
  done
}

#check_modules
main "$@"
