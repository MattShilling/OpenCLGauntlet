#!/bin/bash
#SBATCH -J OpenCL_MULT_ADD
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o main.out
#SBATCH -e main.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shillinm@oregonstate.edu

K1=1
K2=2
K4=4
K8=8
K16=16
K32=32
K64=64
K128=128
K256=256
K512=512
M1=1024
M2=2048
M4=4096
M8=8192

make clean
make

for LOCAL_SIZE in 8 16 32 64 128 256 512
do
        for NUM_E in ${K1} ${K2} ${K4} ${K8} ${K16} ${K32} ${K64} ${K128} ${K256} ${K512} ${M1} ${M2} ${M4} ${M8}
    do 
        ./main a=b*c+d ${NUM_E} ${LOCAL_SIZE}
    done
done