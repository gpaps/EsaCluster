#!/bin/bash

# -------------------------------------
# 🚀 Sweep training runs using Hydra
# -------------------------------------
# Hyperparameter sweep: learning_rate × batch_size
# Each combo runs sequentially (Hydra -m)

echo "Launching sweep..."

# Optional: activate conda environment
source ~/.bashrc
conda activate detectron2

# Move to project directory (adjust path)
cd /home/gpaps/cluster/ship-detectron2/

# Sweep command
python TrainV3_Ships[HydraOnly].py -m \
  training.learning_rate=0.0001,0.00025,0.0005 \
  training.batch_size=4,8,16

echo "Sweep complete."
