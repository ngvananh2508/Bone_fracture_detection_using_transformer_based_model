import os
import torchvision

ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Create dataset class for tranformer-based model like DETR to handle images and annotations automatically
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_dir: str, image_processor, train: bool = True):
        annotation_file_dir = os.path.join(image_dir, ANNOTATION_FILE_NAME)
        super().__init__(image_dir, annotation_file_dir)
        self.image_processor = image_processor
    
    def __getitem__(self, idx):
        images, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target



        