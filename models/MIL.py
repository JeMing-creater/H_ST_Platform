import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(AttentionMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))  # Reshape to (batch_size, num_instances, input_dim)
        # x: (batch_size, num_instances, input_dim)
        H = self.feature_extractor(x)  # (batch_size, num_instances, hidden_dim)
        A = self.attention(H)  # (batch_size, num_instances, 1)
        A = torch.softmax(A, dim=1)  # attention weights
        M = torch.sum(A * H, dim=1)  # (batch_size, hidden_dim)
        out = self.classifier(M)  # (batch_size, 1)
        return out


if __name__ == '__main__':
    import os
    import yaml
    from easydict import EasyDict
    # change to outside directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)
    os.chdir(parent_path)
    print(f"now directiory change to: {os.getcwd()}")
    # Load configuration
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

    input = torch.randn(2, 1, 1024, 1024)
    model = AttentionMIL()
    out = model(input)
    print("Output shape:", out.shape)

