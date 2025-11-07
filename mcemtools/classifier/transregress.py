import torch
import torch.nn.functional as F
import torch.nn as nn

class TransformerRegressor(nn.Module):
    def __init__(self, n_channels=12, n_features=40, d_model=40, nhead=8, num_layers=8, hidden_dim=8192, out_dim=4):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, n_channels, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling and output head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, out_dim)

        # Pooling and confidence head
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.output2 = nn.Linear(d_model, out_dim)

    def forward(self, inp):
        """
        x: (batch, n_channels, n_features)
        returns: (batch, 4)
        """
        # Project input
        x = self.input_proj(inp) + self.pos_embedding  # (batch, 12, d_model)

        # Encode
        x = self.encoder(x)  # (batch, 12, d_model)

        confids = x.detach().clone()
        confids = confids.transpose(1, 2)
        confids = self.pool2(confids).squeeze(-1)
        confids = self.output2(confids)
        
        # Pool across tokens
        x = x.transpose(1, 2)  # (batch, d_model, 12)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        output = self.output(x)

        return output, confids**2

class TransformerLoss(nn.Module):
    def __init__(self, data_gen, classifier_weight = 100, TF_imbalance = 5):
        super(TransformerLoss, self).__init__()
    
    def forward(self, preds, labels, inds):
        clssif, confid = preds
        
        margin_loss_per_output = (clssif - labels)**2
        
        confid_loss = ((confid - margin_loss_per_output.detach().clone())**2).mean()**0.5
        
        margin_loss = (margin_loss_per_output.mean(1)**0.5).mean()

        return (margin_loss, confid_loss)