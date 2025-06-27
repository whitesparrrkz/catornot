import torch.nn as nn
import timm

class CatornotClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CatornotClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        # remove last output layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # the new last layer is now classifier, which will have 2 classes
        # cats, or notcats
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output