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

    def forward(self, x, return_states = False):
        """
        x: (batch, n_channels, n_features)
        returns: (batch, 4)
        """
        # Project input
        x = self.input_proj(x) + self.pos_embedding  # (batch, 12, d_model)

        # Encode
        x = self.encoder(x)  # (batch, 12, d_model)
        if return_states:
            states = x.detach().clone()
        # Pool across tokens
        x = x.transpose(1, 2)  # (batch, d_model, 12)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # Final output
        if return_states:
            return self.output(x), x, states
        return self.output(x), x

# # Activation Recorder for functional activations (F.gelu etc.)
# class ActivationRecorder:
#     def __init__(self):
#         self.records = {}
#         self._orig_fns = {}
#
#     def _record_fn(self, name, fn):
#         def wrapped_fn(*args, **kwargs):
#             out = fn(*args, **kwargs)
#             if name not in self.records:
#                 self.records[name] = []
#             self.records[name].append(out.detach().cpu())
#             return out
#         return wrapped_fn
#
#     def wrap(self):
#         for name in ["gelu", "relu", "sigmoid", "tanh", "softmax"]:
#             if hasattr(F, name):
#                 self._orig_fns[name] = getattr(F, name)
#                 setattr(F, name, self._record_fn(name, getattr(F, name)))
#
#     def unwrap(self):
#         for name, fn in self._orig_fns.items():
#             setattr(F, name, fn)
#         self._orig_fns.clear()
#
# # Utility: Register hooks to capture module activations
# def register_hooks_to_all(model, activations_dict):
#     """
#     Attach forward hooks to all modules that are nonlinear ops
#     like nn.GELU, nn.ReLU, nn.Sigmoid, etc.
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
#             def hook_fn(mod, inp, out, name=name):
#                 if name not in activations_dict:
#                     activations_dict[name] = []
#                 activations_dict[name].append(out.detach().cpu())
#             module.register_forward_hook(hook_fn)
#
# # Modified TransformerRegressor with optional return_states
# class TransformerRegressor(nn.Module):
#     def __init__(self, n_channels=12, n_features=40, d_model=40,
#                  nhead=8, num_layers=8, hidden_dim=8192, out_dim=4):
#         super().__init__()
#
#         self.input_proj = nn.Linear(n_features, d_model)
#         self.pos_embedding = nn.Parameter(torch.randn(1, n_channels, d_model))
#
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=hidden_dim,
#                 activation='gelu',
#                 batch_first=True
#             )
#             for _ in range(num_layers)
#         ])
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.output = nn.Linear(d_model, out_dim)
#
#     def forward(self, x, return_states=False):
#         states = {}
#         x = self.input_proj(x) + self.pos_embedding
#         states["input_proj"] = x.detach().cpu()
#
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             states[f"layer_{i}_output"] = x.detach().cpu()
#
#         x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
#         states["pooled"] = x_pooled.detach().cpu()
#
#         preds = self.output(x_pooled)
#         states["output"] = preds.detach().cpu()
#
#         if return_states:
#             return preds, states
#         else:
#             return preds

class TransformerLoss(nn.Module):
    def __init__(self, data_gen, classifier_weight = 100, TF_imbalance = 5):
        super(TransformerLoss, self).__init__()
        self.label_mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, preds, labels, inds):
        clssif = preds[0]
        
        batch_size = len(clssif)
        
        margin_loss_all = ((labels - clssif)**2).mean((1))**0.5
        margin_loss = margin_loss_all.mean()

        return (margin_loss, margin_loss_all.detach(), margin_loss_all.detach())