import os
import json
import torch
import warnings
import hydra
import ray
from omegaconf import DictConfig
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data import transforms as T
from detectron2 import model_zoo
from register_dataset import register_dataset

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.ResizeShortestEdge(short_edge_length=(720, 1024), sample_style="choice"),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.3),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.2, horizontal=False, vertical=True),
                T.RandomRotation(angle=[-30, 30]),
                T.RandomCrop("relative_range", (0.85, 0.85))
            ]),
        )


@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(cfg: DictConfig):
    print(f"\n🚀 Starting training with:\n"
          f"  Dataset: {cfg.dataset.name}\n"
          f"  LR: {cfg.training.learning_rate}\n"
          f"  Batch Size: {cfg.training.batch_size}\n"
          f"  Model: {cfg.model.architecture}\n")

    ray.init(ignore_reinit_error=True)
    register_dataset()

    with open(cfg.dataset.json_path, 'r') as f:
        coco_data = json.load(f)
    num_classes = len(coco_data["categories"])

    detectron_cfg = get_cfg()
    detectron_cfg.merge_from_file(model_zoo.get_config_file(cfg.model.architecture))
    detectron_cfg.DATASETS.TRAIN = (cfg.dataset.name,)
    detectron_cfg.DATASETS.TEST = ()

    detectron_cfg.DATALOADER.NUM_WORKERS = cfg.training.num_workers
    detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.model.architecture)
    detectron_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    detectron_cfg.SOLVER.IMS_PER_BATCH = cfg.training.batch_size
    detectron_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.training.batch_size_per_image
    detectron_cfg.SOLVER.BASE_LR = cfg.training.learning_rate
    detectron_cfg.SOLVER.MAX_ITER = cfg.training.max_iterations
    detectron_cfg.SOLVER.STEPS = tuple(cfg.training.steps)
    detectron_cfg.SOLVER.GAMMA = 0.1
    detectron_cfg.SOLVER.AMP.ENABLED = True
    detectron_cfg.SOLVER.CHECKPOINT_PERIOD = cfg.training.checkpoint_period

    detectron_cfg.MODEL.ROI_HEADS.NMS_THRESH_TRAIN = cfg.training.nms_thresh_train
    detectron_cfg.INPUT.MIN_SIZE_TRAIN = tuple(cfg.training.min_size_train)
    detectron_cfg.INPUT.MIN_SIZE_TEST = cfg.training.min_size_test
    detectron_cfg.INPUT.RANDOM_FLIP = "horizontal"

    detectron_cfg.SOLVER.WEIGHT_DECAY = cfg.training.weight_decay
    detectron_cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = cfg.training.pre_nms_topk_train
    detectron_cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = cfg.training.post_nms_topk_train
    detectron_cfg.MODEL.ROI_HEADS.FOCAL_LOSS_GAMMA = cfg.training.focal_loss_gamma
    detectron_cfg.MODEL.ROI_HEADS.FOCAL_LOSS_ALPHA = cfg.training.focal_loss_alpha

    detectron_cfg.OUTPUT_DIR = cfg.output.dir
    os.makedirs(cfg.output.dir, exist_ok=True)

    trainer = AugmentedTrainer(detectron_cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ray.shutdown()


if __name__ == "__main__":
    main()
