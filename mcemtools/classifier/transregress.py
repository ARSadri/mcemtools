import torch
import torch.nn.functional as F
import torch.nn as nn

class TransformerRegressor(nn.Module):
    def __init__(self, n_channels=12, n_features=40, d_model=40, nhead=8, 
                       num_layers=8, hidden_dim=8192, out_dim=4):
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

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, out_dim)

        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.output2 = nn.Linear(d_model, out_dim)

    def forward(self, inp):
        x = self.input_proj(inp) + self.pos_embedding  # (batch, 12, d_model)

        x = self.encoder(x)  # (batch, 12, d_model)

        confids = x.detach().clone()
        confids = confids.transpose(1, 2)
        confids = self.pool2(confids).squeeze(-1)
        confids = self.output2(confids)
        
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
    
class TransformerRegressor_with_recycling(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_features=40,
        d_model=40,
        nhead=8,
        num_layers=8,
        hidden_dim=8192,
        out_dim=4,
        n_recycles=3, #3 for AlphaFold and 7 for RNN
        detach_recycle=True, #True for AlphaFold and False for RNN
    ):
        super().__init__()
        assert n_recycles > 0
        self.n_channels = n_channels
        self.out_dim = out_dim
        self.n_recycles = n_recycles
        self.detach_recycle = detach_recycle

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_channels, d_model)
        )

        self.recycle_proj = nn.Linear(out_dim, d_model)
        self.recycle_gate = nn.Parameter(torch.zeros(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, out_dim)

        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.output2 = nn.Linear(d_model, out_dim)

    def _single_pass(self, inp, prev_estimate):
        recycle = self.recycle_proj(prev_estimate)
        recycle = recycle[:, None, :].expand(-1, self.n_channels, -1)

        x = self.input_proj(inp)
        x = x + self.pos_embedding
        x = x + self.recycle_gate.tanh()[None, None, :] * recycle

        x = self.encoder(x)

        reconst = self.decoder(x).squeeze(-1)

        conf = x.detach()
        conf = conf.transpose(1, 2)
        conf = self.pool2(conf).squeeze(-1)
        conf = self.output2(conf)

        x_pool = x.transpose(1, 2)
        x_pool = self.pool(x_pool).squeeze(-1)
        pred = self.output(x_pool)

        return pred, conf**2, reconst

    def forward(self, inp, init_estimate=None):
        B = inp.shape[0]

        if init_estimate is None:
            pred = torch.zeros(
                B, self.out_dim, device=inp.device, dtype=inp.dtype,)
        else:
            pred = init_estimate

        confid = None

        for _ in range(self.n_recycles):
            if self.detach_recycle:
                pred = pred.detach()

            pred, confid, reconst = self._single_pass(inp, pred)

        return pred, confid, reconst

class TransformerLoss_with_recycling(nn.Module):
    def __init__(self, data_gen, classifier_weight = 0.75):
        super(TransformerLoss_with_recycling, self).__init__()
        self.data_gen = data_gen
        self.classifier_weight = classifier_weight
    
    def forward(self, preds, labels, inds):

        clssif, confid, reconst = preds
        
        margin_loss_per_output = (clssif - labels)**2
        
        confid_loss = ((confid - margin_loss_per_output.detach().clone())**2).mean()**0.5
        
        margin_loss = (margin_loss_per_output.mean(1)**0.5).mean()

        data, labels = self.data_gen(inds)
        #the data looks like n_samples, n_pixels, n_features
        #we know that we would like to reconstruct data[:, :, :40].sum(-1)
        # as such the loss should look like 
        reconst_loss = (((reconst - data[:, :, :40].mean(-1))**2).mean(1)**0.5).mean()
        total_loss = margin_loss * self.classifier_weight + reconst_loss * (1 - self.classifier_weight)

        return (total_loss, confid_loss)