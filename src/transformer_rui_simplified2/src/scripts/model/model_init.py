from transformer.CrossSectionalAttention import CrossSectionalSelfAttention
from transformer.TransformerEncoder import TransformerEncoder
from transformer.PredictionHeadReturns import PredictionHeadReturns
from transformer.PredictionHeadRegime import PredictionHeadRegime
from transformer.TemporalSelfAttention import TemporalSelfAttention
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from tqdm import tqdm


#from src.scripts.transformer.Dummy import DummyPredictor


class ModelPipeline(nn.Module):
    def __init__(self, config):
        super(ModelPipeline, self).__init__()

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

        self.predictor_returns = PredictionHeadReturns(
            d_model=config['d_model'],
            output_dim=config['output_dim'],
        )
        self.predictor_regime = PredictionHeadRegime(
            d_model=config['d_model'],
            output_dim=config['output_dim'],
        )

        self.wrapper = self._set_wrapper(config)

        self.alpha = config['alpha']
        self.zeta = config['zeta']
        self.norm = config['norm']
        self.loss_fn_returns = config['loss_method']

        self.loss_fn_regime = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

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
            x: Tensor of shape [B, T, S, F] (Batch, Time, Symbols, Features)

        Returns:
            pred_returns: [B, S] — regression output
            pred_regime: [B, S] — classification output (logits)
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
        pred_returns, pred_regime = self.predictor_returns(x), self.predictor_regime(x)

        return pred_returns, pred_regime  # [B, S]

    def to(self, device):
        super().to(device)
        return self

    def train_mode(self):
        self.train()
        if self.wrapper is not None:
            self.wrapper.train()
        self.encoder.train()
        self.predictor_returns.train()
        self.predictor_regime.train()

    def eval_mode(self):
        self.eval()
        if self.wrapper is not None:
            self.wrapper.eval()
        self.encoder.eval()
        self.predictor_returns.eval()
        self.predictor_regime.eval()

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

    def evaluate_epoch(self, dataloader, device, track_best=False):
        """
        Evaluate on the test set.
        """
        self.eval_mode()
        total_loss = 0.0
        all_returns = []
        all_regimes = []
        all_return_targets = []
        all_regime_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(device)  # [B, T, S, F]
                targets = targets.to(device)  # [B, S, 2]

                try:
                    out_returns, out_regime = self.forward(inputs)  # both [B, S]
                except Exception as e:
                    print(f"[ERROR] Forward pass failed during evaluation: {e}")
                    continue

                if (
                        torch.isnan(out_returns).any() or torch.isnan(out_regime).any() or
                        torch.isinf(out_returns).any() or torch.isinf(out_regime).any()
                ):
                    print("[ERROR] Outputs contain NaN or Inf during evaluation")
                    continue

                target_return = targets[:, :, 0]  # [B, S]
                target_regime = targets[:, :, 1].float()  # [B, S]

                loss_return = self.loss_fn_returns(out_returns, target_return)
                loss_regime = self.loss_fn_regime(out_regime, target_regime)

                if (
                        torch.isnan(loss_return).any() or torch.isnan(loss_regime).any() or
                        torch.isinf(loss_return).any() or torch.isinf(loss_regime).any()
                ):
                    print("[ERROR] Loss is NaN or Inf during evaluation")
                    continue

                loss = self.alpha * loss_return + (1 - self.alpha) * loss_regime
                total_loss += loss.item()

                all_returns.append(out_returns.cpu())
                all_regimes.append(out_regime.cpu())
                all_return_targets.append(target_return.cpu())
                all_regime_targets.append(target_regime.cpu())

        avg_loss = total_loss / max(len(dataloader), 1)

        # Track best model
        if track_best and avg_loss < self.best_test_loss:
            self.best_test_loss = avg_loss

            self.best_model_state = {
                'encoder': self.encoder.state_dict(),
                'predictor_returns': self.predictor_returns.state_dict(),
                'predictor_regime': self.predictor_regime.state_dict(),
            }

            if self.wrapper is not None:
                self.best_model_state['wrapper'] = self.wrapper.state_dict()

            # Combine predictions and targets per task
            self.best_predictions = {
                'returns': torch.cat(all_returns, dim=0),
                'regimes': torch.cat(all_regimes, dim=0)
            }
            self.best_targets = {
                'returns': torch.cat(all_return_targets, dim=0),
                'regimes': torch.cat(all_regime_targets, dim=0)
            }

        return avg_loss, self.best_predictions, self.best_targets

    def train_epoch(self, dataloader, device):
        self.train_mode()
        total_loss = 0.0
        batch_count = 1

        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(device)  # [B, T, S, F]
            targets = targets.to(device)  # [B, S, 2]

            self.optimizer.zero_grad()

            try:
                pred_returns, pred_regime = self.forward(inputs)  # [B, S], [B, S]
            except Exception as e:
                print(f"[ERROR] Forward pass failed: {e}")
                continue

            # Split target into two components
            target_return = targets[:, :, 0]  # [B, S]
            target_regime = targets[:, :, 1].float()  # [B, S]


            # Compute each loss separately
            loss_return = self.loss_fn_returns(pred_returns, target_return)
            loss_regime = self.loss_fn_regime(pred_regime, target_regime)

            # Combine with alpha weight (self.alpha) + ridge regularization
            loss = (
                    self.alpha * loss_return
                    + (1 - self.alpha) * loss_regime
                    + self.zeta * sum(p.norm(self.norm) ** self.norm for p in self.parameters() if p.requires_grad)
            )

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            except RuntimeError as e:
                print(f"[ERROR] Backward pass failed: {e}")
                continue

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / max(batch_count - 1, 1)

        return avg_train_loss
