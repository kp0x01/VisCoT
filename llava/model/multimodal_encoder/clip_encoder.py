import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.ft_vision_tower = getattr(args, "ft_vision_tower", False)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self._force_fp32 = True  # Force fp32 for vision tower
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        
        # Load the vision tower in fp32
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            torch_dtype=torch.float32
        )
        
        # Force convert to fp32
        self.vision_tower = self.vision_tower.to(torch.float32)

        if not self.ft_vision_tower:
            self.vision_tower.requires_grad_(False)
        else:
            self.vision_tower.requires_grad_(True)

        self.is_loaded = True

    def _apply(self, fn):
        """Override _apply to prevent unwanted dtype conversions"""
        # This is called by .to(), .cuda(), .half(), etc.
        # We want to block half() conversions for the vision tower
        
        # Apply to everything except vision_tower if we're forcing fp32
        super()._apply(fn)
        
        # Force vision tower back to fp32 if it was changed
        if self.is_loaded and self._force_fp32:
            # Check if vision tower is not fp32
            if self.vision_tower.dtype != torch.float32:
                print(f"[CLIPVisionTower] Forcing vision_tower back to fp32 (was {self.vision_tower.dtype})")
                self.vision_tower = self.vision_tower.to(torch.float32)
        
        return self

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        # Ensure vision tower is in fp32 before forward pass
        if self.is_loaded and self.vision_tower.dtype != torch.float32:
            print(f"[CLIPVisionTower] Vision tower dtype mismatch detected: {self.vision_tower.dtype}, converting to fp32")
            self.vision_tower = self.vision_tower.to(torch.float32)
            
        if self.ft_vision_tower:
            return self.forward_func(images)
        else:
            with torch.no_grad():
                return self.forward_func(images)

    def forward_func(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Always return fp32 when forcing it
        if self._force_fp32:
            return torch.float32
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2