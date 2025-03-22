# EsaCluster


# 🚢 Ship Detection with Detectron2 + Hydra

This project trains a Faster R-CNN model on ship detection datasets using [Detectron2](https://github.com/facebookresearch/detectron2), [Hydra](https://hydra.cc/), and optional [Ray](https://docs.ray.io/) support. It's designed for scalable training on both local machines and HPC clusters.

---

## 📁 Project Structure

your_project/ ├── TrainV3_Ships[HydraOnly].py # Main training script ├── config/ │ └── train_config.yaml # All training configs (Hydra-compatible) ├── data/ │ ├── Final_Fully_Cleaned_Ships.json │ └── _Optical-Ship/ │ └── VHRShips_ShipRSImageNEt/ ├── register_dataset.py # Handles dataset registration ├── trained_models/ # Where output models are saved └── outputs/ # Hydra-run logs and configs


---

## ⚙️ Configuration (Hydra)

All training parameters (dataset paths, hyperparameters, etc.) are defined in:

📄 `config/train_config.yaml`

Modify this file to:
- Switch datasets
- Tune hyperparameters
- Change output locations

Example fields:
```yaml
training:
  batch_size: 4
  learning_rate: 0.00025
  max_iterations: 80000
  steps: [20000, 50000, 70000]
