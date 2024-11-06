#!/bin/bash 
#! this code is used to run wandb agent on specified GPUs and CPUs, zelin 2024-2-16

# wandb

# Check if at least three arguments were provided (script name, agent ID, and GPU list are always present)
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <agent_id> <gpu_list> <cpus_per_program>"
    echo "Example: $0 AGENT_ID '0,1,2' 2"
    exit 1
fi

# The first argument is the agent ID
AGENT_ID=$1

# The second argument is the list of GPUs
GPU_LIST=$2

# The third argument is the number of CPUs per program
CPUS_PER_PROGRAM=$3

# Split the GPU list into an array using comma as a delimiter
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Initialize the starting CPU ID
CPU_START=0

# Launch wandb agents on specified CUDA devices and CPUs
for GPU_ID in "${GPUS[@]}"
do
    echo "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
    echo "Starting wandb agent $AGENT_ID on GPU $GPU_ID with $CPUS_PER_PROGRAM CPUs"
    echo "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
    # Calculate the CPU end index based on the number of CPUs per program
    let CPU_END=CPU_START+CPUS_PER_PROGRAM-1

    # Use taskset with a range of CPUs
    CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CPU_START-$CPU_END wandb agent $AGENT_ID &

    # Update the starting CPU ID for the next program
    let CPU_START+=CPUS_PER_PROGRAM

    sleep 30
done

# Wait for all background processes to finish
wait
