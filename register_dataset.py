import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Paths
dataset_name = "Ships_Optical"
json_path = "/home/gpaps/PycharmProject/ESA/Data/JSON files/1_Ships/Final_Fully_Cleaned_Ships[VHRShips_ShipsImageNet].json"
image_root = "/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/VHRShips_ShipRSImageNEt/"

def register_dataset():
    """
    Register the dataset if not already registered.
    """
    if dataset_name in MetadataCatalog.list():
        print(f"Dataset {dataset_name} already registered.")
        return

    assert os.path.exists(json_path), f"JSON file not found: {json_path}"
    assert os.path.exists(image_root), f"Image directory not found: {image_root}"

    register_coco_instances(dataset_name, {}, json_path, image_root)
    print(f" Successfully registered dataset: {dataset_name}")

if __name__ == "__main__":
    register_dataset()
