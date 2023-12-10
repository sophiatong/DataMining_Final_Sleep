
#!/usr/bin/env python3

import os
import sys
import argparse
import random
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from SleepDataLoaders import SleepDataModule, SleepDataSet
from SleepModel import SleepModel
from torchsummary import summary
import time
import math
import concurrent.futures as cf
from itertools import repeat
import pandas as pd

def make_filename(stage, ids, window_size, sliding_step_size, downsample_val, time_to_next_cut, batch_size, imbalance_factor, num_augment, permute_col, rep=None, drop=""):
    return f"data_cache/{stage}_reduc_{'_'.join(map(str, ids))}_{window_size}_{sliding_step_size}_{downsample_val}_{time_to_next_cut}_{batch_size}_{imbalance_factor}_{num_augment}_p{permute_col}_{rep}{drop}"


def main():
    # define constants
    seed = 1
    random.seed(seed)
    SORTED_IDS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    N_PATIENTS = len(SORTED_IDS)
    shuffled_ids = random.sample(SORTED_IDS, N_PATIENTS)

    # parameters
    window_size = 40
    sliding_step_size = 15
    batch_size = 16
    downsample_size = 200
    time_to_next_cut = 30
    imbalance_factor = 1
    num_augment = 5
    learning_rate = 5e-8
    test_size = 0.2
    n_signals = 18 - 3 ## TODO
    drop_cols = [9, 11, 12]
    n_repeats = 5
    drop_str = "_drop9_11_12"

    # ids in train test split
    n_patients_test = math.floor(test_size * N_PATIENTS)
    test_ids = shuffled_ids[:n_patients_test]
    train_ids = shuffled_ids[n_patients_test:]

    signal_num_list = []
    rep_list = []
    for signal_num in range(n_signals):
        for rep in range(n_repeats):
            signal_num_list.append(signal_num)
            rep_list.append(rep)

    with cf.ThreadPoolExecutor() as executor:
        executor.map(make_test_data_permutations, 
                     signal_num_list,
                     rep_list,
                     repeat(test_ids),
                     repeat(window_size),
                     repeat(sliding_step_size),
                     repeat(downsample_size),
                     repeat(time_to_next_cut),
                     repeat(batch_size),
                     repeat(imbalance_factor),
                     repeat(num_augment),
                     repeat(drop_cols),
                     repeat(drop_str)
        )

    # make train test datasets
    train_file_path = make_filename("train", train_ids, window_size, sliding_step_size, downsample_size, 
                                    time_to_next_cut, batch_size, imbalance_factor, num_augment, None, drop="drop_str")
    train_dataset = SleepDataSet(
        train_ids, window_size, sliding_step_size, downsample_size, time_to_next_cut, 
        train_file_path, imbalance_factor, num_augment, None, drop_cols=drop_cols
    )
    test_file_path = make_filename("test", test_ids, window_size, sliding_step_size, downsample_size, 
                                    time_to_next_cut, batch_size, imbalance_factor, num_augment, None, drop="drop_str")
    test_dataset = SleepDataSet(
        test_ids, window_size, sliding_step_size, downsample_size, time_to_next_cut, 
        test_file_path, imbalance_factor, num_augment, None, drop_cols=drop_cols
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # # instantiate and train TODO
    mod = SleepModel(n_signals=n_signals, signal_length=downsample_size, lr=learning_rate)
    logger = TensorBoardLogger("tb_logs", name=f"prod{drop_str}")
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10, save_last=True, dirpath="model_checkpoints"
    )
    trainer = L.Trainer(
        max_epochs=25, logger=logger, enable_progress_bar=True, 
        callbacks=[ModelSummary(max_depth=3), checkpoint_callback]
    )
    try:
        mod = SleepModel.load_from_checkpoint(f"model_checkpoints/final_mod{drop_str}", 
                                              n_signals=n_signals, signal_length=downsample_size)
        print("Model loaded successfully from checkpoint")
    except FileNotFoundError:
        print("Checkpoint file not found. Training model:")
        start = time.time()
        trainer.fit(mod, train_dataloaders=train_dataloader)
        trainer.save_checkpoint(f"model_checkpoints/final_mod{drop_str}")
        end = time.time()
        total_time_min = round((end - start) / 60, 2)
        print(f"  Finished trainer.fit in {total_time_min} mins")
    # test model    
    test_results = trainer.test(mod, dataloaders=test_dataloader)
    # store test results
    test_loss_list = [test_results[0]['test_loss']]
    test_acc_list = [test_results[0]['test_acc']]
    test_f1_list = [test_results[0]['test_f1']]
    test_roc_list = [test_results[0]['test_roc_auc']]
    perm_col_list = [None]
    perm_rep_list = [None]

    # permute test set and predict
    for signal_num in range(n_signals):
        for rep in range(n_repeats):
            perm_test_file_path = make_filename("perm_test", test_ids, window_size, sliding_step_size, downsample_size, 
                                                time_to_next_cut, batch_size, imbalance_factor, num_augment, signal_num, rep, drop_str)
            perm_test_dataset = SleepDataSet(
                test_ids, window_size, sliding_step_size, downsample_size, time_to_next_cut, 
                perm_test_file_path, imbalance_factor, num_augment, signal_num, rep, drop_cols
            )
            perm_test_dataloader = DataLoader(perm_test_dataset, batch_size=batch_size)
            print(f"TESTING --- PERMUTE COL {signal_num}, repetition {rep}")
            test_results = trainer.test(mod, dataloaders=perm_test_dataloader) 
            results_dic = test_results[0]
            test_loss_list.append(results_dic['test_loss'])
            test_acc_list.append(results_dic['test_acc'])
            test_f1_list.append(results_dic['test_f1'])
            test_roc_list.append(results_dic['test_roc_auc'])
            perm_col_list.append(signal_num)
            perm_rep_list.append(rep)
    
    out_metrics = pd.DataFrame({
        "permuted_col": perm_col_list,
        "permutation_repeat_no": perm_rep_list,
        "test_loss": test_loss_list,
        "test_roc_auc": test_roc_list,
        "test_f1": test_f1_list,
        "test_acc": test_acc_list
    })
    out_metrics.to_csv(f"out_metrics{drop_str}.csv", 
                       index=False)


def make_test_data_permutations(signal_num, rep, test_ids, window_size, sliding_step_size, downsample_size, 
                                time_to_next_cut, batch_size, imbalance_factor, num_augment, drop_cols, drop_str):
    perm_test_file_path = make_filename("perm_test", test_ids, window_size, sliding_step_size, downsample_size, 
                                        time_to_next_cut, batch_size, imbalance_factor, num_augment, signal_num, rep, drop_str)
    perm_test_dataset = SleepDataSet(
    test_ids, window_size, sliding_step_size, downsample_size, time_to_next_cut, 
    perm_test_file_path, imbalance_factor, num_augment, signal_num, rep, drop_cols
    )
    print(f"  DONE perm {signal_num}, rep {rep}")


if __name__ == '__main__':
    main()