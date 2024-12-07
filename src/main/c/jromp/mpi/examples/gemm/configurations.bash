#!/bin/bash

function configurations {
  local nodes_conf=(1 5 16)    # Incremented by one, because the master occupies one additional task, so one more node is needed
  local n_tasks_conf=(2 16 31) # -1 for master, so 1, 15, 30 (divisors of the matrix size)
  local cpus_per_task_conf=(1 12 32)
  local optimization_level_conf=(0 1 2 3)

  echo "Configurations:"
  echo "  Nodes: ${nodes_conf[*]}"
  echo "  Processes: ${n_tasks_conf[*]}"
  echo "  Threads per process: ${cpus_per_task_conf[*]}"
  echo "  Optimization level: ${optimization_level_conf[*]}"

  echo " ****** Total combinations: $((${#nodes_conf[@]} * ${#n_tasks_conf[@]} * ${#cpus_per_task_conf[@]} * ${#optimization_level_conf[@]})) ******"
}
