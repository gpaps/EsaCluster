import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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
        self.cfg = cfg

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
        self.logger.info(f"📏 MAE (bbox center): {mae_center:.2f} pixels")

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

            # Write per-run CSV
            if self.output_dir:
                csv_path = os.path.join(self.output_dir, "eval_metrics.csv")
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Metric", "Value"])
                    writer.writerow(["bbox_center_mae", round(mae_center, 2)])
                    for cls, val in per_class_ap50.items():
                        writer.writerow([f"AP50_{cls}", val])
                self.logger.info(f"Saved metrics to: {csv_path}")

            # Append to global metrics_summary.csv
            try:
                run_info = {
                    "learning_rate": self.cfg.training.learning_rate,
                    "batch_size": self.cfg.training.batch_size,
                    "mae": round(mae_center, 2),
                }
                run_info.update({f"AP50_{cls}": val for cls, val in per_class_ap50.items()})

                global_csv = os.path.join(self.cfg.OUTPUT_DIR, "metrics_summary.csv")
                write_header = not os.path.exists(global_csv)
                with open(global_csv, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=run_info.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(run_info)
                self.logger.info(f"Appended metrics to global summary: {global_csv}")
            except Exception as e:
                self.logger.warning(f"Failed to write global summary: {e}")

            # Save bar chart
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(per_class_ap50.keys(), per_class_ap50.values(), color="steelblue")
                ax.set_title("AP@50 per Class")
                ax.set_ylabel("AP@50")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45)
                plot_path = os.path.join(self.output_dir, "ap50_per_class.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                self.logger.info(f"Saved AP50 bar chart to: {plot_path}")
            except Exception as e:
                self.logger.warning(f"Failed to generate AP50 plot: {e}")

        return results
