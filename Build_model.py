import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

class DeTr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, idslabel, TRAIN_DATALOADER, VAL_DATALOADER):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model_name_or_path="facebook/detr-resnet-50",
                                                            num_labels=len(idslabel),
                                                            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.TRAIN_DATALOADER = TRAIN_DATALOADER
        self.VAL_DATALOADER = VAL_DATALOADER
    
    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    # common steps in forward path for both training and validation
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"training_{k}", v.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss
    
    def configure_optimizers(self):

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
        
    def train_dataloader(self):
        return self.TRAIN_DATALOADER
    
    def val_dataloader(self):
        return self.VAL_DATALOADER
    
    



