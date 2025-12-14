# Bone fracture detection using transformer-based model
This project implements end-to-end object detection using DEtection TRanformer (DETR) model from Hugging Face Transformers to detect bone fractures in X-ray images.


DETR treats object detection as a direct set prediction problem, eliminating the need for hand-crafted components like anchor boxes or non-maximum suppression. The model is fine-tuned on a bone fracture dataset (typically in COCO format) to localize fractures with bounding boxes and with pytorch lightning for simplified, scalable training. The data was downloaded from Roboflow Bone Fracture Dataset.
