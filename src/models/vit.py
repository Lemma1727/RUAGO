from torch import nn
from transformers import ViTConfig, ViTModel

class ViT(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ViT, self).__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            vit_config = ViTConfig() # default
            self.vit = ViTModel(vit_config)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x.last_hidden_state[:, 0, :])
        return x