import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCNNEncoder(nn.Module):
    def __init__(self, kernel_size=5, stride=1, feature_dim= 192, dim_out= 2048):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 100, kernel_size= kernel_size, stride= stride, padding=0)  # causal pad time
        self.conv2 = nn.Conv2d(100, 100, kernel_size= kernel_size, stride= stride, padding=0)
        self.conv3 = nn.Conv2d(100, 64, kernel_size= kernel_size, stride= stride, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size= kernel_size, stride= stride, padding=0)

        self.bn1 = nn.BatchNorm2d(100)
        self.bn2 = nn.BatchNorm2d(100)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * feature_dim, dim_out)
        )

    def forward(self, x):  # x: [B, 1, T, F]
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))  # (left, right, top, bottom)
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv2(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv3(x))
        x = torch.nn.functional.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv4(x))
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # [B, T, 64*F]
        x = self.fc(x)  # [B, T, dim_out]
        return x  # [B, T, dim_out]

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.se(x)  
        return x * se  



class GlobalCNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, dilation, n_dropout=0.0):
        super(GlobalCNNBlock, self).__init__()
        # Point-wise CNN 1
        self.pw_cnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size_pw)

        # Dilated Depth-wise CNN
        self.dw_cnn = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size_dw,
            dilation=dilation,
            groups= hidden_dim,
            padding= dilation * (kernel_size_dw - 1) // 2  # this auto keeps T
        )

        # Point-wise CNN 2
        self.pw_cnn2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size_pw)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(input_dim)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)

        # ReLU
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(n_dropout)

        self.kernel_size_dw = kernel_size_dw
        self.dilation = dilation

    def forward(self, x):
        # x: [B, T, D] -> Chuyển thành [B, D, T] cho Conv1d
        x = x.transpose(1, 2)  # [B, D, T]
        residual = x

        # print(f"x.shape: {x.shape}")
        # Point-wise CNN 1
        x = self.pw_cnn1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # print(f"x.shape: {x.shape}")
        # Dilated Depth-wise CNN (padding thủ công để giữ nguyên chiều dài)
        # pad = (self.kernel_size_dw - 1) * self.dilation  # Padding bên trái để đảm bảo nhân quả
        # x = F.pad(x, (pad, 0))  # Chỉ pad bên trái
        x = self.dw_cnn(x)
        x = self.relu(x)
        x = self.bn2(x)
        # print(f"x.shape: {x.shape}")

        # Point-wise CNN 2
        x = self.pw_cnn2(x)
        x = self.bn3(x)

        # print(f"x.shape: {x.shape}")
        # Squeeze and Excitation
        x = self.se(x)

        # print(f"x.shape: {x.shape}")
        # Dropout
        x = self.dropout(x)

        # Residual Connection
        x = x + residual

        # print(f"x.shape: {x.shape}")

        # Chuyển lại về [B, T, D]
        x = x.transpose(1, 2)
        return x
    
class GlobalCNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, n_layers=6, n_dropout=0.0):
        super(GlobalCNNEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GlobalCNNBlock(input_dim, hidden_dim, kernel_size_pw, kernel_size_dw, dilation= 2**i, n_dropout= n_dropout) 
            for i in range(1, n_layers+1)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self, local_cnn, global_cnn, d_input, d_output):
        super(CNNEncoder, self).__init__()
        self.local_cnn = local_cnn
        self.global_cnn = global_cnn
        self.projected = nn.Linear(d_input * 2, d_output)  
    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        x = x.unsqueeze(1)
        local_out = self.local_cnn(x)  # [B, 64, T, F]
        # print(f"local_out.shape: {local_out.shape}")
        global_out = self.global_cnn(local_out)  # [B, T, 64*F]
        # print(f"global_out.shape: {global_out.shape}")
        concat = torch.cat([local_out, global_out], dim=2)  # [B, T, 128*F]
        # print(f"concat.shape: {concat.shape}")
        
        output = self.projected(concat)  # [B, T, 128*F] -> [B, T, d_input]
        # print(output.shape)
        # exit()
        return output
    
def build_cnn_encoder(config):
    local_cnn = LocalCNNEncoder(
        kernel_size= config["local_cnn_encoder"]["kernel_size"],
        stride= config["local_cnn_encoder"]["stride"], 
        feature_dim= config["local_cnn_encoder"]["feature_dim"],
        dim_out= config["local_cnn_encoder"]["dim_out"]
    )
    global_cnn = GlobalCNNEncoder(
        input_dim= config["global_cnn_encoder"]["input_dim"], 
        hidden_dim= config["global_cnn_encoder"]["hidden_dim"],
        kernel_size_pw= config["global_cnn_encoder"]["kernel_size_pw"],
        kernel_size_dw= config["global_cnn_encoder"]["kernel_size_dw"],
        n_layers= config["global_cnn_encoder"]["n_layers"],
        n_dropout= config["global_cnn_encoder"]["n_dropout"]
    )
    return CNNEncoder(local_cnn, global_cnn, d_input= config["global_cnn_encoder"]["input_dim"], d_output= config["dim_out"])
