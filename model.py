import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torchvision.models.resnet import resnet50 as _resnet50


class ReidResNet(nn.Module):
    def __init__(self, num_classes=500, dropout=0):
        super().__init__()
        resnet = _resnet50(pretrained=True)
        out_planes = resnet.fc.in_features
        modules = [
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(out_planes),
            nn.Dropout(dropout),
        ]
        self.extractor = nn.Sequential(*modules)

        self.hooks = None
        for _, module in self.extractor.named_modules():
            if isinstance(module, Flatten):
                module.register_forward_hook(self._hook_fn)

        self.classifier = nn.Linear(out_planes, num_classes)

    def forward(self, x):
        x = self.extractor(x)
        prob = self.classifier(x)
        return prob

    def _hook_fn(self, module, input, output):
        self.hooks = output.cpu().detach()


if __name__ == "__main__":
    model = ReidResNet()
    data = torch.randn(100, 3, 128, 64)
    out = model(data)
    print(model.hooks.shape)
    print(out.shape)
