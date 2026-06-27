import torch
import torch.nn as nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNLSTMModel(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 num_layers=2,
                 dropout=0.3):
        super().__init__()

        backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.cnn_out_size = 1792  

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  #biLSTM
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)        
        features = self.cnn(x)             
        features = features.view(B*T, -1)  
        features = features.view(B, T, -1) 

        lstm_out, _ = self.lstm(features)  

        last = lstm_out[:, -1, :]          

        return self.classifier(last)       


class CNNTransformerModel(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=4,
                 dropout=0.3):
        super().__init__()


        backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out_size = 1792

        self.input_proj = nn.Linear(self.cnn_out_size, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 5, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)                     
        feat = feat.view(B * T, -1)           
        feat = feat.view(B, T, -1)            
        feat = self.input_proj(feat)          

        feat = feat + self.pos_embedding      

        cls  = self.cls_token.expand(B, -1, -1)  
        feat = torch.cat([cls, feat], dim=1)      

        out = self.transformer(feat)          

        cls_out = out[:, 0, :]                

        return self.classifier(cls_out)       

def build_model(model_name='cnn_lstm'):
    if model_name == 'cnn_lstm':
        model = CNNLSTMModel()
    elif model_name == 'transformer':
        model = CNNTransformerModel()
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model
