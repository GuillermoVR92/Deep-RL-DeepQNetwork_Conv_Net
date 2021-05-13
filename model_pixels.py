import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkBasic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=256, fc4_units=256, fc5_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkBasic, self).__init__()
        self.seed = torch.manual_seed(seed)
               
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)        
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        # finally, create action_size output channel
        self.fc5 = nn.Linear(fc4_units, action_size)
                

    def forward(self, state):
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        # final output
        return x

class QNetworkFull(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkFull, self).__init__()
        self.seed = torch.manual_seed(seed)
        n_filter_1 = 128
        n_filter_2 = 256
        n_filter_3 = 256
        # NHWC (Batch, Height, Width, Channels)

        self.conv_layer_1 = nn.Conv3d(in_channels=3, out_channels=n_filter_1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.maxpool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.batch_norm_1 = nn.BatchNorm3d(n_filter_1)
        
        self.conv_layer_2 = nn.Conv3d(in_channels=n_filter_1, out_channels=n_filter_2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.batch_norm_2 = nn.BatchNorm3d(n_filter_2)
        
        self.conv_layer_3 = nn.Conv3d(in_channels=n_filter_2, out_channels=n_filter_3, kernel_size=(4, 2, 2), stride=(1, 2, 2))
        self.batch_norm_3 = nn.BatchNorm3d(n_filter_3)
        
        conv_net_output_size = self._get_conv_out_size(state_size)

        self.fully_connected_1 = nn.Linear(conv_net_output_size, 1024)     
        # finally, create action_size output channel
        self.fully_connected_2 = nn.Linear(1024, action_size)

    def _compose_Conv_Net(self, input_state):
        
        x = F.relu(self.conv_layer_1(input_state))
        x = self.maxpool_1(x)
        x = self.batch_norm_1(x)

        x = F.relu(self.conv_layer_2(x))
        x = self.batch_norm_2(x)

        x = F.relu(self.conv_layer_3(x))
        x = self.batch_norm_3(x)

        x = x.view(x.size(0), -1)

        return x

    # generate random sample to get size of Conv Net
    def _get_conv_out_size(self, shape):
            x = torch.rand(shape)
            x = self._compose_Conv_Net(x)
            n_size = x.data.view(1, -1).size(1)
            return n_size
        

    def forward(self, state):
        
        # two linear layers
        x = self._compose_Conv_Net(state)
        x = F.relu(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        
        # final output
        return x
    
