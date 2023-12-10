import lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

class SleepModule(nn.Module):
    def __init__(self, n_signals, signal_length):
        super().__init__()
        self.n_signals = n_signals
        self.signal_length = signal_length
        # cnn blocks
        self.conv_block1 = self._conv_block(self.n_signals, 32, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = self._conv_block(32, 32, kernel_size=29, stride=1, padding=14)
        self.conv_block3 = self._conv_block(32, 64, kernel_size=29, stride=1, padding=14)

        # transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128, dropout=0.3, batch_first=True)
        self.transf_enc_block = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # prediction_block
        self.prediction_block = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.signal_length),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

        # Initialize weights using Xavier initialization
        self.apply(self._initialize_weights)

    def forward(self, inputs, target):
        # apply blocks sequentially
        #print(f"in - {inputs.shape}")
        assert(bool(torch.isinf(inputs).any()) == False)
        assert(bool(torch.isinf(target).any()) == False)
        x = torch.transpose(inputs, 1, 2)
        #print(f"in_transpose - {x.shape}")
        x = self.conv_block1(x)
        #print(f"conv1 - {x.shape}")
        x = self.conv_block2(x)
        #print(f"conv2 - {x.shape}")
        x = self.conv_block3(x)
        #print(f"conv3 - {x.shape}")

        x = torch.transpose(x, 1, 2)
        #print(f"conv3_transpose - {x.shape}")
        x = self.transf_enc_block(x)
        #print(f"transf_enc - {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"trans_enc_transpose - {x.shape}")
        y = self.prediction_block(x)
        #print(f"pred - {y.shape}")

        y = y.to(dtype=torch.float32).squeeze()
        #print(f"pred_sqeeuze - {y.shape}")
        assert(bool(torch.isinf(y).any()) == False)
        return y
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def _initialize_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    

class SleepModel(L.LightningModule):
    def __init__(self, n_signals, signal_length, lr=1e-4):
        super().__init__()
        self.mymodel = SleepModule(n_signals, signal_length)
        self.learning_rate = lr

    def forward(self, inputs, target):
        return self.mymodel(inputs, target)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.mymodel(inputs, target)
        # n_positive = target.sum() + 1 # add value so not to divide by 0
        # n_negative = len(target) - n_positive
        # pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32, device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss() # pos_weight=pos_weight)
        output = output.to(self.device)
        target = target.to(self.device)
        loss = criterion(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        predicted_probs = torch.sigmoid(output) # we need this here as our loss contains sigmoid layer
        roc_auc = roc_auc_score(target.cpu().numpy(), predicted_probs.cpu().detach().numpy())
        roc_auc = torch.tensor(roc_auc, dtype=torch.float32)
        self.log("train_roc_auc", roc_auc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        predicted_labels = (predicted_probs > 0.5).float() # we cut at 0.5 just as a baseline
        acc = accuracy_score(target.cpu().numpy(), predicted_labels.cpu().numpy())
        acc = torch.tensor(acc, dtype=torch.float32)
        f1 = f1_score(target.cpu().numpy(), predicted_labels.cpu().numpy(), zero_division=0.0)
        f1 = torch.tensor(f1, dtype=torch.float32)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.mymodel(inputs, target)
        #n_positive = target.sum() + 1
        #n_negative = len(target) - n_positive
        #pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32, device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss() # pos_weight=pos_weight)
        output = output.to(self.device)
        target = target.to(self.device)
        loss = criterion(output, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        predicted_probs = torch.sigmoid(output) # we need this here as our loss contains sigmoid layer
        roc_auc = roc_auc_score(target.cpu().numpy(), predicted_probs.cpu().numpy())
        roc_auc = torch.tensor(roc_auc, dtype=torch.float32)
        self.log("val_roc_auc", roc_auc, on_epoch=True, prog_bar=True, logger=True)

        predicted_labels = (predicted_probs > 0.5).float() # we cut at 0.5 just as a baseline
        acc = accuracy_score(target.cpu().numpy(), predicted_labels.cpu().numpy())
        acc = torch.tensor(acc, dtype=torch.float32)
        f1 = f1_score(target.cpu().numpy(), predicted_labels.cpu().numpy(), zero_division=0.0)
        f1 = torch.tensor(f1, dtype=torch.float32)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.mymodel(inputs, target)
        # n_positive = target.sum() + 1
        # n_negative = len(target) - n_positive
        # pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32, device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss()#pos_weight=pos_weight)
        output = output.to(self.device)
        target = target.to(self.device)
        loss = criterion(output, target)
        self.log("test_loss", loss)

        predicted_probs = torch.sigmoid(output) # we need this here as our loss contains sigmoid layer
        try:
            roc_auc = roc_auc_score(target.cpu().numpy(), predicted_probs.cpu().numpy())
            roc_auc = torch.tensor(roc_auc, dtype=torch.float32)
            self.log("test_roc_auc", roc_auc, on_epoch=True, prog_bar=True, logger=True)
        except ValueError:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")
            self.log("test_roc_auc", np.nan, on_epoch=True, prog_bar=True, logger=True)
        
        predicted_labels = (predicted_probs > 0.5).float() # we cut at 0.5 just as a baseline
        acc = accuracy_score(target.cpu().numpy(), predicted_labels.cpu().numpy())
        acc = torch.tensor(acc, dtype=torch.float32)
        f1 = f1_score(target.cpu().numpy(), predicted_labels.cpu().numpy(), zero_division=0.0)
        f1 = torch.tensor(f1, dtype=torch.float32)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch):
        inputs, target = batch
        return self.mymodel(inputs, target)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.mymodel.parameters(), lr=self.learning_rate)
    
