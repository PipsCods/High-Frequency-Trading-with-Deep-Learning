from transformer.CrossSectionalAttention import CrossSectionalSelfAttention
from transformer.CustomLoss import CustomLoss
from transformer.TransformerEncoder import TransformerEncoder
from transformer.PredictionHead import PredictionHead
from transformer.TemporalSelfAttention import TemporalSelfAttention
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from tqdm import tqdm


#from src.scripts.transformer.Dummy import DummyPredictor


class ModelPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._validation_config(config)  # Confirm that the config is complete

        self.baseline_attention = config.get('baseline_attention', 'time')  # Get baseline attention

        # Get positioning of categorical and continuous variables
        self.cat_feat_positions = config['cat_feat_positions']
        self.cont_feat_positions = config['cont_feat_positions']

        self.encoder = TransformerEncoder(
            vocab_sizes=config['vocab_sizes'],
            embed_dims=config['embed_dims'],
            num_cont_features=config['num_cont_features'],
            d_model=config['d_model'],
            seq_len=config['seq_len'],
            num_layers=config['num_layers'],
            expansion_factor=config['expansion_factor'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],

        )  # Instantiate the encoder

        #self.predictor = DummyPredictor(constant_value=0.02)

        self.predictor = PredictionHead(
            d_model=config['d_model'],
            output_dim=config['output_dim'],
        )

        self.wrapper = self._set_wrapper(config)

        self.loss_method = config['loss_method']
        self.alpha = config['alpha']
        self.loss_fn = self._set_loss_fn()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'], weight_decay=1e-2)

        total_steps = config["total_steps"]

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # Track best model
        self.best_test_loss = float('inf')
        self.best_model_state = None
        self.best_predictions = None
        self.best_targets = None

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [B, T, S, D_total]
                    - D_total = num_cat_features + num_cont_features

        Returns:
            Tensor of shape [B, S] — final predictions (after squeezing)
        """
        # Slice features

        cat_inputs = inputs[..., self.cat_feat_positions]
        cat_inputs = cat_inputs.long()  # [B, T, S, num_cat_feats]
        cont_inputs = inputs[..., self.cont_feat_positions]  # [B, T, S, num_cont_feats]
        # Embedding + projection
        emb = self.encoder.embedding(cat_inputs)  # [B, T, S, emb_dim]
        x = self.encoder.projector(emb, cont_inputs)  # [B, T, S, d_model]
        x = self.encoder.pos_encoder(x)  # [B, T, S, d_model]

        # Apply baseline attention
        if self.baseline_attention == 'cross-sectional':
            x = self._cross_attention(x)  # [B, T, S, d_model]
        elif self.baseline_attention == 'time':
            x = self._time_attention(x)  # [B, T, S, d_model]
        else:
            raise ValueError(f"Unknown baseline attention mode: {self.baseline_attention}")

        # Optional wrapper
        if self.wrapper is not None:
            x = self.wrapper(x)  # [B, T, S, d_model]

        # Predict using last time step
        x = x[:, -1, :, :]  # [B, S, d_model]
        output = self.predictor(x)  # [B, S, 1]

        return output.squeeze(-1)  # [B, S]

    def to(self, device):
        # self.encoder.to(device)
        # if self.wrapper is not None:
        #     self.wrapper.to(device)
        # self.predictor.to(device)
        super().to(device)
        return self

    def train_mode(self):
        self.train()
        if self.wrapper is not None:
            self.wrapper.train()
        self.predictor.train()
        self.encoder.train()

    def eval_mode(self):
        self.eval()
        if self.wrapper is not None:
            self.wrapper.eval()
        self.predictor.eval()
        self.encoder.eval()

    def _validation_config(self, config):
        """
        Parameters validation
        """
        assert config['d_model'] % config['n_heads'] == 0, "d_model must be divisible by n_heads"

        required_params = ["vocab_sizes", 'embed_dims', 'num_cont_features',
                           'continuous_variables', 'cat_feat_positions', 'cont_feat_positions', 'd_model',
                           'seq_len', 'num_layers', 'expansion_factor', 'n_heads', 'dropout', 'output_dim', 'lr',
                           'loss_method', 'wrapper', 'alpha', 'baseline_attention', 'total_steps']

        if not all(param in config for param in required_params):
            raise ValueError(f"Parameters missing, required: {required_params}")

        if config['dropout'] < 0 or config['dropout'] > 1:
            raise ValueError("Dropout must be between 0 and 1")


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

    def _set_wrapper(self, config):
        wrapper = config['wrapper']

        if wrapper == 'cross-sectional':
            wrapper = CrossSectionalSelfAttention(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                dropout=config['dropout'],
            )
        elif wrapper == 'time':
            wrapper = TemporalSelfAttention(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                dropout=config['dropout'],
            )

        else:
            wrapper = None

        return wrapper

    def _set_loss_fn(self):
        x = self.loss_method
        if x == 'mse':
            return nn.MSELoss()
        elif x == 'custom':
            return CustomLoss(alpha=self.alpha)
        elif x == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function not recognized")

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
                inputs = inputs.to(device)  # [B, T, S, features]
                targets = targets.to(device)  # [B, S]

                try:
                    outputs = self.forward(inputs)  # should be [B, S]
                except Exception as e:
                    print(f"[ERROR] Forward pass failed during evaluation: {e}")
                    continue

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("[ERROR] Outputs contain NaN or Inf during evaluation")
                    continue

                loss = self.loss_fn(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[ERROR] Loss is NaN or Inf during evaluation")
                    continue

                total_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_tgts.append(targets.cpu())

        avg_loss = total_loss / max(len(dataloader), 1)

        if track_best and avg_loss < self.best_test_loss:
            self.best_test_loss = avg_loss

            self.best_model_state = {
                'encoder': self.encoder.state_dict(),
                'predictor': self.predictor.state_dict(),
            }

            if self.wrapper is not None:
                self.best_model_state['wrapper'] = self.wrapper.state_dict()

            self.best_predictions = torch.cat(all_preds, dim=0)
            self.best_targets = torch.cat(all_tgts, dim=0)

        return avg_loss, self.best_predictions, self.best_targets

    def train_epoch(self, dataloader, device):
        self.train_mode()
        total_loss = 0.0
        batch_count = 1

        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(device)  # [B, T, S, features]
            targets = targets.to(device)  # [B, S]

            self.optimizer.zero_grad()

            try:
                outputs = self.forward(inputs)  # should be [B, S] consistently
            except Exception as e:
                print(f"[ERROR] Forward pass failed: {e}")
                continue


            loss = self.loss_fn(outputs, targets)

            try:
                loss.backward()
            except RuntimeError as e:
                print(f"[ERROR] Backward pass failed: {e}")
                continue  # Don't call .item() if backward fails

            """# Gradient sanity check
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"[ERROR] Gradient has NaN or Inf in {name}")
                        raise ValueError("Invalid gradient detected")
                else:
                    print(f"[WARNING] No gradient for parameter: {name}")"""

            #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / max(batch_count - 1, 1)
        return avg_train_loss
