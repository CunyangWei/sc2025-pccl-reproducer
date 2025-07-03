#!/bin/bash

#SBATCH --account bega-delta-gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpuA100x4
#SBATCH --time=00:05:00
#SBATCH --output=raw_collectives_benchmark.out
#SBATCH --error=raw_collectives_benchmark.err


module load nccl
module load cuda/12.6.3
source /u/cunyang/pccl-venv/bin/activate
module load PrgEnv-gnu cray-mpich

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 


## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export MPICH_GPU_SUPPORT_ENABLED=1

# disabling rendezvous mode
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 

# mapping GPUs to NICs and processes correctly
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:3; 1:2; 2:1; 3:0"
export CUDA_VISIBLE_DEVICES=3,2,1,0
export CPU_MASK="--cpu-bind=cores"


# make FI output useful information
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_VERBOSE=1
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1

# collecting counter data
#export MPICH_OFI_CXI_COUNTER_REPORT=5

# SCRIPT="python -u benchmark_raw_collectives/all_to_all.py \
SCRIPT="python -u benchmark_raw_collectives/all_gather.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --library pccl --test"


# for JIT building
export CXX=CC 
export CC=cc
export PYTHONPATH="$PYTHONPATH:."

chmod +x /u/cunyang/sc2025-pccl-reproducer/scripts/get_rank.sh
run_cmd="srun --mpi=pmi2 -C gpu -N $NNODES -n $GPUS $CPU_MASK --gpus-per-node=4 ./u/cunyang/sc2025-pccl-reproducer/scripts/get_rank.sh $SCRIPT"
echo $run_cmd 
eval $run_cmd

