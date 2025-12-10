import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_xavier(m):
    '''
    Applies Xavier (Glorot) initialization to the layers of a model.
    
    Usage:
        model = MyModel()
        model.apply(init_weights_xavier)
    '''
    # Check if the module is an instance of a convolutional or linear layer
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # Apply Xavier uniform initialization to the weights
        nn.init.xavier_uniform_(m.weight)
        
        # Initialize biases to a small constant value (e.g., 0.01) or zero.
        # Initializing to a small constant can sometimes help break symmetry,
        # but zero is also a common and safe choice.
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class FiLMLayer(nn.Module):
    '''
    Feature-wise Linear Modulation layer
    This layer takes a feature map `x` and applies a transformation
    to it, conditioned on parameters `gamma` and `beta`
    
    output = gamma * x + beta
    '''
    def forward(self, x, gamma, beta):
        # Reshape gamma and beta for broadcasting over the sequence length
        # x.shape:      [batch_size, channels, sequence_length]
        # gamma.shape:  [batch_size, channels] -> [batch_size, channels, 1]
        # beta.shape:   [batch_size, channels] -> [batch_size, channels, 1]
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class FiLMGenerator(nn.Module):
    '''
    An MLP that generates the `gamma` and `beta` parameters for all FiLM layers
    in the network, based on the scalar physical inputs.
    '''
    def __init__(self, num_mlp_inputs, total_film_channels):
        super(FiLMGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(num_mlp_inputs, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            # The output layer must produce 2 values (gamma, beta) for each channel
            # across all FiLM'd layers.
            nn.Linear(128, total_film_channels * 2) 
        )

    def forward(self, scalar_input):
        return self.generator(scalar_input)


class InceptionModule1D(nn.Module):
    '''
    The Inception module from the 'InceptCurves' paper, adapted for 1D data.
    It applies multiple convolutions of different kernel sizes in parallel and concatenates their outputs.
    This is based on the diagram in Figure 4 of the paper.
    '''
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule1D, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm1d(n1x1),
            nn.ReLU(True),
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm1d(n3x3_reduce),
            nn.ReLU(True),
            nn.Conv1d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(n3x3),
            nn.ReLU(True),
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm1d(n5x5_reduce),
            nn.ReLU(True),
            nn.Conv1d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(n5x5),
            nn.ReLU(True),
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        # Process input through all four parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate the outputs along the channel dimension (dim=1)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptCurvesFiLM(nn.Module):
    def __init__(self, num_scalars, num_outputs):
        super(InceptCurvesFiLM, self).__init__()

        # Moderate reduction throughout
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=7, stride=2, padding=3),  # 32→24
            nn.BatchNorm1d(24),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.inception1 = InceptionModule1D(in_channels=24, n1x1=12, n3x3_reduce=18, n3x3=24, n5x5_reduce=3, n5x5=6, pool_proj=6)  # ~25% reduction
        self.film1_channels = 12 + 24 + 6 + 6  # 48 (was 64)
        
        self.inception2 = InceptionModule1D(in_channels=self.film1_channels, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)  # ~25% reduction
        self.film2_channels = 48 + 96 + 24 + 24  # 192 (was 256)
        
        self.inception3 = InceptionModule1D(in_channels=self.film2_channels, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)
        self.inception4 = InceptionModule1D(in_channels=192, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.film_layer1 = FiLMLayer()
        self.film_layer2 = FiLMLayer()

        total_film_channels = self.film1_channels + self.film2_channels  # 48 + 192 = 240
        self.film_generator = FiLMGenerator(num_scalars, total_film_channels)

        # Proportional regressor
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(192, 96),  # ~60% reduction
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(96, num_outputs)
        )

    def forward(self, seabed_input, scalar_input):
        if seabed_input.dim() == 2:
            # Input shape: [batch_size, seq_len] -> add channel dimension
            seabed_input = seabed_input.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Generate all FiLM parameters from scalar inputs
        # film_params shape: [batch_size, (64*2 + 256*2)] = [batch_size, 640]
        film_params = self.film_generator(scalar_input)
        
        # Split the generated parameters for each FiLM layer
        # The split sizes must be (gamma1_size, beta1_size, gamma2_size, beta2_size, ...)
        gamma1, beta1, gamma2, beta2 = torch.split(film_params, 
                                                   [self.film1_channels, self.film1_channels, 
                                                    self.film2_channels, self.film2_channels], dim=1)
        
        # Process seabed profile and apply FiLM modulation 
        x = self.pre_conv(seabed_input)
        
        # Block 1 + FiLM 1
        x = self.inception1(x)
        x = self.pool(x)
        x = self.film_layer1(x, gamma1, beta1)
        
        # Block 2 + FiLM 2
        x = self.inception2(x)
        x = self.pool(x)
        x = self.film_layer2(x, gamma2, beta2)

        # Remaining blocks (unmodulated)
        x = self.inception3(x)
        x = self.pool(x)
        x = self.inception4(x)

        # Final prediction from modulated features
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        output = self.regressor(x)
        
        return output