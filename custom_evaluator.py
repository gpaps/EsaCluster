import os
import csv
import numpy as np
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

class ShipDetectionEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self.logger = setup_logger(name="ShipDetectionEvaluator")
        self.predicted_boxes = []
        self.gt_boxes = []
        self.image_ids = []
        self.output_dir = output_dir

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt_instances = input["instances"]
            gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
            pred_boxes = output["instances"].pred_boxes.tensor.cpu().numpy()

            self.gt_boxes.extend(gt_boxes)
            self.predicted_boxes.extend(pred_boxes)
            self.image_ids.append(input["image_id"])

        super().process(inputs, outputs)

    def evaluate(self):
        results = super().evaluate()

        if not self.predicted_boxes or not self.gt_boxes:
            self.logger.warning("No predictions or ground truths collected for MAE calculation.")
            return results

        count = min(len(self.gt_boxes), len(self.predicted_boxes))
        gt = np.array(self.gt_boxes[:count])
        preds = np.array(self.predicted_boxes[:count])

        gt_centers = (gt[:, :2] + gt[:, 2:]) / 2
        pred_centers = (preds[:, :2] + preds[:, 2:]) / 2
        mae_center = np.mean(np.linalg.norm(gt_centers - pred_centers, axis=1))

        results["bbox_center_mae"] = {"MAE(px)": mae_center}
        self.logger.info(f"MAE (bbox center): {mae_center:.2f} pixels")

        # Save metrics to CSV
        if self._eval_predictions:
            classwise_results = self._eval_predictions["bbox"]["precision"]
            class_names = self._metadata.get("thing_classes", [])

            # Collect per-class AP50 values
            per_class_ap50 = {}
            for idx, cls in enumerate(class_names):
                ap50 = classwise_results[0, :, idx, 0, 2]
                valid = ap50[ap50 > -1]
                mean_ap50 = np.mean(valid) * 100 if len(valid) > 0 else -1
                per_class_ap50[cls] = round(mean_ap50, 2)

            results["per_class_ap50"] = per_class_ap50

            # Write CSV
            if self.output_dir:
                csv_path = os.path.join(self.output_dir, "eval_metrics.csv")
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Metric", "Value"])
                    writer.writerow(["bbox_center_mae", round(mae_center, 2)])
                    for cls, val in per_class_ap50.items():
                        writer.writerow([f"AP50_{cls}", val])
                self.logger.info(f"📁 Saved metrics to: {csv_path}")

        return results