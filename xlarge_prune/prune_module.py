import math
import torch
import torch.nn as nn
import torch.nn.init as init

class MaskBlock(nn.Module):
    def __init__(self, out_features):
        super(MaskBlock, self).__init__()
        self.extractor = nn.Linear(out_features, 2 * out_features, bias=True)
        self.projector = nn.Linear(2 * out_features, out_features, bias=True)

        # Initialize weights for extractor layer (ReLU activation)
        init.kaiming_uniform_(self.extractor.weight, mode='fan_in', nonlinearity='relu')
        if self.extractor.bias is not None:
            init.zeros_(self.extractor.bias)

        # Initialize weights for projector layer (Sigmoid activation)
        init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            init.zeros_(self.projector.bias)

    def forward(self, x):
        hidden_state = torch.relu(self.extractor(x)).t()
        mask = torch.sigmoid(self.projector(hidden_state.t())).t()
        return mask



class Projected_MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(Projected_MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Define multiple mask blocks
        self.num_blocks = 2
        self.blocks = nn.ModuleList([MaskBlock(out_features) for _ in range(self.num_blocks)])
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
        # Move all parameters to the appropriate device
        self.to(device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        mask = self.weight  
        for block in self.blocks:
            mask = block(mask.t())
        masked_weight = self.weight * mask
        return nn.functional.linear(input, masked_weight, self.bias)
