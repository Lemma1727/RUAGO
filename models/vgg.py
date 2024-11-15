from torch import nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16(nn.Module):
    def __init__(self, vgg_name='VGG16', num_classes=10):
        super(VGG16, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = nn.AvgPool2d(out.size()[3],1)(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
    
    def get_features(self, x):
        out = self.features(x)
        out = nn.AvgPool2d(out.size()[3],1)(out)
        out = out.view(out.size(0), -1)
        return out

    def forward_with_features(self, x):
        out = self.features(x)
        out = nn.AvgPool2d(out.size()[3],1)(out)
        out = out.view(out.size(0), -1)
        return out, self.classifier(out)
    
if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16()
    model.to(device)
    
    dummy_input = torch.randn(20, 3, 224, 224, device=device)
    output = model(dummy_input)
    print(f"Forward Output Shape: {output.shape}")

    features = model.get_features(dummy_input)
    print(f"Extracted Features Shape: {features.shape}")