import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_dataset():
    """
    Register training and optional validation dataset.
    """
    train_name = "Ships_Optical"
    val_name = "Ships_Optical_val"

    base_path = os.getcwd()  # allows Hydra to work with relative paths
    json_train = os.path.join(base_path, "Final_Fully_Cleaned_Ships[VHRShips_ShipsImageNEt].json")
    img_train = os.path.join(base_path, "_Optical-Ship/VHRShips_ShipRSImageNEt/")

    json_val = os.path.join(base_path, "Final_Fully_Cleaned_Ships[VHRShips_ShipsImageNEt].json")
    img_val = img_train  # assuming images are shared

    if train_name not in MetadataCatalog.list():
        register_coco_instances(train_name, {}, json_train, img_train)
        print(f"Registered training set: {train_name}")
    else:
        print(f"Training set {train_name} already registered.")

    if os.path.exists(json_val):
        if val_name not in MetadataCatalog.list():
            register_coco_instances(val_name, {}, json_val, img_val)
            print(f" Registered validation set: {val_name}")
        else:
            print(f"Validation set {val_name} already registered.")
    else:
        print(f" Validation JSON not found at: {json_val}")
