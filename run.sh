export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3
torchrun \
  --nproc_per_node 2 \
  --master_port 29550 \
  train_Survival_Prediction.py