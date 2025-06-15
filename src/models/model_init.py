from ..transformer.CrossSectionalAttention import CrossSectionalSelfAttention
from ..transformer.CustomLoss import CustomLoss
from ..transformer.TransformerEncoder import TransformerEncoder
from ..transformer.PredictionHead import PredictionHead
from ..transformer.TemporalSelfAttention import TemporalSelfAttention
# from ..transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from tqdm import tqdm

class ModelPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._validation_parameters(config)

        self.encoder = TransformerEncoder(
            basic_embed_dims=config['basic_embed_dims'],
            embed_dims=config['embed_dims'],
            vocab_sizes_basic=config['vocab_sizes_basic'],
            vocab_sizes=config['vocab_sizes'],
            num_cont_features=config['num_cont_features'],
            d_model=config['d_model'],
            seq_len=config['seq_len'],
            num_layers=config['num_layers'],
            expansion_factor=config['expansion_factor'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
        )

        self.attention_mode = config['initial_attention']

        if config['wrapper'] == 'cross-sectional':
            self.wrapper = CrossSectionalSelfAttention(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                dropout=config['dropout'],
            )
        elif config['wrapper'] == 'time':
            self.wrapper = TemporalSelfAttention(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                dropout=config['dropout'],
            )

        else:
            self.wrapper = None

        #self.predictor = DummyPredictor(constant_value=0.02)

        self.predictor = PredictionHead(
            d_model=config['d_model'],
            output_dim=config['output_dim'],
        )

        if config['loss_method'] == 'mse':
            self.loss_fn = nn.MSELoss()
        elif config['loss_method'] == 'custom':
            self.loss_fn = CustomLoss(alpha= config["alpha"])
        elif config['loss_method'] == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function not recognized")

        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'], weight_decay=1e-2)
        total_steps = config["total_steps"]
        # self.scheduler = get_linear_schedule_with_warmup(
        #         self.optimizer,
        #         num_warmup_steps=int(0.1 * total_steps),
        #         num_training_steps=total_steps,
        #     )
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'], weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )

        # Track best model
        self.best_test_loss = float('inf')
        self.best_model_state = None
        self.best_predictions = None
        self.best_targets = None

        # Position of the additional features in the input tensor
        self.cat_feat_positions = config['cat_feat_positions']
        self.cont_feat_positions = config['cont_feat_positions']

    def forward(self, inputs):
        # Slice categorical and continuous features
        cat_inputs = inputs[..., self.cat_feat_positions]  # [B, T, S, num_cat_feats]
        cont_inputs = inputs[..., self.cont_feat_positions]  # [B, T, S, num_cont_feats]

        # Embedding and projecting
        emb = self.encoder.embedding(cat_inputs)  # [B, T, S, emb_dim]
        x = self.encoder.projector(emb, cont_inputs)  # [B, T, S, d_model]
        x = self.encoder.pos_encoder(x)  # [B, T, S, d_model]

        # Apply the main attention mode
        if self.attention_mode == 'cross-sectional':
            x = self._cross_attention(x)  # returns [B, T, S, D]
        elif self.attention_mode == 'time':
            x = self._time_attention(x)  # returns [B, T, S, D]
        else:
            raise ValueError("Attention mode not recognized")

        # Apply wrapper
        if self.wrapper is not None:
            x = self.wrapper(x)  # [B, T, S, D]

        # Predict final output from last time step
        output = self.predictor(x)  # [B, S] or [B, S, 1]
        return output


    def to(self, device):  
        # self.encoder.to(device)
        # if self.wrapper is not None:
        #     self.wrapper.to(device)
        # self.predictor.to(device) 
        super().to(device)
        return self

    def parameters(self, **kwargs):
        out = list(self.encoder.parameters())
        if self.wrapper is not None:
            out += list(self.wrapper.parameters())
        if any(p.requires_grad for p in self.predictor.parameters()):
            out += list(self.predictor.parameters())
        return out

    def train_mode(self):
        self.encoder.train()
        if self.wrapper is not None:
            self.wrapper.train()
        self.predictor.train()

    def eval_mode(self):
        self.encoder.eval()
        if self.wrapper is not None:
            self.wrapper.eval()
        self.predictor.eval()

    def _validation_parameters(self, config):
        # Validation of parameters

        assert config['d_model'] % config['n_heads'] == 0, "d_model must be divisible by n_heads"

        required_params = ['basic_embed_dims', 'embed_dims', 'vocab_sizes_basic',
                           'vocab_sizes', 'num_cont_features', 'd_model', 'seq_len',
                           'num_layers', 'expansion_factor', 'n_heads', 'dropout', 'wrapper']

        if not all(param in config for param in required_params):
            raise ValueError(f"Parameters missing, required: {required_params}")

        if config['dropout'] < 0 or config['dropout'] > 1:
            raise ValueError("Le dropout doit être entre 0 et 1")

    def _cross_attention(self, x):
        B, T, S, D = x.shape
        x = x.reshape(B * T, S, D)  # attention across stocks at each time step

        for block in self.encoder.blocks:
            x = block(x)  # [B*T, S, D]

        x = x.reshape(B, T, S, D)  # back to [B, T, S, D]
        return x

    def _time_attention(self, x):
        B, T, S, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)  # attention across time for each stock

        for block in self.encoder.blocks:
            x = block(x)  # [B*S, T, D]

        x = x.reshape(B, S, T, D).permute(0, 2, 1, 3)  # back to [B, T, S, D]
        return x


    def evaluate_epoch(self, dataloader, device, track_best=False):
        """
        Evaluate on the test set.
        """
        self.eval_mode()
        total_loss = 0.0
        all_preds = []
        all_tgts = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(device)  # inputs shape: [B, T, S, features]
                targets = targets.to(device)  # targets shape: [B, S] or [B, S, 1]

                outputs = self.forward(inputs)

                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_tgts.append(targets.cpu())

        avg_loss = total_loss / len(dataloader)

        if track_best and avg_loss < self.best_test_loss:
            self.best_test_loss = avg_loss

            if self.wrapper is not None:
                self.best_model_state = {
                    'encoder': self.encoder.state_dict(),
                    'predictor': self.predictor.state_dict(),
                    'wrapper': self.wrapper.state_dict(),
                }

            else:
                self.best_model_state = {
                    'encoder': self.encoder.state_dict(),
                    'predictor': self.predictor.state_dict(),
                }

            self.best_predictions = torch.cat(all_preds)
            self.best_targets = torch.cat(all_tgts)

        return avg_loss, self.best_predictions, self.best_targets

    def train_epoch(self, dataloader, device):
        self.train_mode()
        total_loss = 0.0
        batch_count = 1

        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(device)  # inputs shape: [B, T, S, features]
            targets = targets.to(device)  # targets shape: [B, S] or [B, S, 1]

            self.optimizer.zero_grad()
            outputs = self.forward(inputs)  # outputs shape: [B, S] or [B, S, 1]
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            grads_norms = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grads_norms[name] = param.grad.data.norm(2).item()
                # if param.grad is None:
                #     print(f"No grad for {name}")
                # elif torch.all(param.grad == 0):
                #     print(f"Zero grad for {name}")
            sorted_grads = sorted(grads_norms.items(), key=lambda x: x[1])
            # tqdm.write(f"[Batch {batch_count+1}/{len(dataloader)}] loss = {loss.item():.4f}")
            # tqdm.write(f"Smallest grad : {sorted_grads[0]}")
            # tqdm.write(f"Largest grad  : {sorted_grads[-1]}")
            batch_count += 1

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            

        avg_train_loss = total_loss / len(dataloader)
        return avg_train_loss