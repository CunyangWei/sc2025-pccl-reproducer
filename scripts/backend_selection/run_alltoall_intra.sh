#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --time=00:20:00
#SBATCH --qos=regular


module load nccl/2.24.3
module load cudatoolkit/12.4
source $SCRATCH/pccl-venv/bin/activate
module load PrgEnv-gnu cray-mpich craype-accel-nvidia80
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export CPATH=/global/common/software/nersc9/nccl/2.24.3/include:$CPATH
export LIBRARY_PATH=/global/common/software/nersc9/nccl/2.24.3/lib:$LIBRARY_PATH
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

# export NCCL_DEBUG=WARN
# make FI output useful information
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_VERBOSE=1
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1

# collecting counter data
#export MPICH_OFI_CXI_COUNTER_REPORT=5

# for JIT building
export CXX=CC 
export CC=cc
export PYTHONPATH="$PYTHONPATH:."

SCRIPT0="python -u benchmark_raw_collectives/all_to_all.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --pccl-disable-cpp-backend \
        --library xccl"

SCRIPT1="python -u benchmark_raw_collectives/all_to_all.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --pccl-disable-cpp-backend \
        --library mpi"


SCRIPT2="python -u benchmark_raw_collectives/all_to_all.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --library xccl"

SCRIPT3="python -u benchmark_raw_collectives/all_to_all.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --library mpi"

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 $CPU_MASK --gpus-per-node=4 ./scripts/get_rank.sh $SCRIPT0"
echo $run_cmd 
eval $run_cmd

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 $CPU_MASK --gpus-per-node=4 ./scripts/get_rank.sh $SCRIPT1"
echo $run_cmd 
eval $run_cmd

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 $CPU_MASK --gpus-per-node=4 ./scripts/get_rank.sh $SCRIPT2"
echo $run_cmd 
eval $run_cmd

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 $CPU_MASK --gpus-per-node=4 ./scripts/get_rank.sh $SCRIPT3"
echo $run_cmd 
eval $run_cmd