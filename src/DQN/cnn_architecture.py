# src/DQN/cnn_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Definition of a Residual Block used for the CNN architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer with batch normalization and ReLU activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)
        # Define the shortcut connection to match the dimensions when needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Forward pass through the first convolution, dropout, and ReLU activation
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        # Forward pass through the second convolution
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection and apply ReLU activation
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Custom CNN architecture extending the BaseFeaturesExtractor from Stable Baselines3
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # Initial convolutional layer and max pooling
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Define a sequence of residual blocks
        self.res_block1 = ResidualBlock(64, 128, stride=2)
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(128, 256, stride=2)
        self.res_block4 = ResidualBlock(256, 256)
        # Global average pooling and fully connected layer to obtain the final feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, features_dim)

    def forward(self, observations):
        # Forward pass through initial convolutional layer and pooling
        x = F.relu(self.conv1(observations))
        x = self.pool(x)
        # Forward pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # Global average pooling and flattening
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        # Forward pass through the fully connected layer with ReLU activation
        x = F.relu(self.fc(x))
        return x
