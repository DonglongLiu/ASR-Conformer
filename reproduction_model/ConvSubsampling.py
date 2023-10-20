import torch
import torch.nn as nn

class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2dSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1)
        conv1 = self.conv1(inputs)
        relu_1 = self.relu(conv1)
        conv2 = self.conv2(relu_1)
        outputs = self.relu(conv2)
        
        batch_size, channels, lengths, dim = outputs.size()
        
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, lengths, channels * dim)
        
        outputs_lengths = input_lengths / 4
        outputs_lengths -= 1
        
        return outputs, outputs_lengths
    

if __name__ == "__main__":
    test = Conv2dSubsampling(1, 3)
    t = torch.randn(3,32,32)
    l = torch.randn(3) # time?
    print('t:')
    print(t)
    print('l:')
    print(l)
    outputs, lengths = test(t, l)
    print('outputs')
    print(outputs)
    print(outputs.size())