#!/bin/bash
#SBATCH -J CudaMonteCarlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o cuda_carlo.out
#SBATCH -e cuda_carlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shillinm@oregonstate.edu

1K=1
2K=2
4K=4
8K=8
16K=16
32K=32
64K=64
128K=128
256K=256
512K=512
1M=1024
2M=2048
4M=4096
8M=8192

make

for LOCAL_SIZE in 8 16 32 64 128 256 512
do
    for NUM_E in ${1K} ${2K} ${4K} ${8K} ${16K} ${32K} ${64K} ${128K} ${256K} ${512K} ${1M} ${2M} ${4M} ${8M}
    do 
        ./main a=b*c ${NUM_E} ${LOCAL_SIZE}
    done
done

for LOCAL_SIZE in 8 16 32 64 128 256 512
do
    for NUM_E in ${1K} ${2K} ${4K} ${8K} ${16K} ${32K} ${64K} ${128K} ${256K} ${512K} ${1M} ${2M} ${4M} ${8M}
    do 
        ./main a=b*c+d ${NUM_E} ${LOCAL_SIZE}
    done
done

for LOCAL_SIZE in 32 64 128 256
do
    for NUM_E in ${1K} ${2K} ${4K} ${8K} ${16K} ${32K} ${64K} ${128K} ${256K} ${512K} ${1M} ${2M} ${4M} ${8M}
    do 
        ./main a:=b*c ${NUM_E} ${LOCAL_SIZE}
    done
done