import random
import pandas as pd
import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import time

DATA_DIR = "rafael data download/merged"
SLEEP_EPOCH_S = 30
SORTED_IDS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

class SleepDataSet(Dataset):
    def __init__(self, patient_ids, window_size, sliding_step_size, downsample=None, time_to_next_cut=45, 
                 cache_path=None, imbalance_factor=1, num_augment=0, permute_feature=None, perm_seed=None, drop_cols=None):
        print(f"  Starting SleepDataSet __init__ for {len(patient_ids)} patients")
        start = time.time()
        self.downsample_val = downsample
        self.time_to_next_cut = time_to_next_cut

        if cache_path is not None:
            try:
                print(f"Attempting to load processed data from {cache_path}")  
                self.load_processed_data(cache_path)
                print(f"Successfully loaded processed data from {cache_path}")
            except FileNotFoundError:
                print(f"Cache file not found. Processing data and saving to {cache_path}")
                X_list_1s = []
                y_list_1s = []
                X_list_0s = []
                y_list_0s = []
                num_original_1s = 0
                for pid in patient_ids:
                    merge_df, respevt_df = self._load_data_and_merge(pid, permute_feature, perm_seed, drop_cols)
                    merge_df = self._trim_merge_df(merge_df)
                    start_sec = merge_df.index[0]
                    print(f"reading patient {pid}")
                    while (start_sec + window_size) < merge_df.index[-1]:
                        window_df, _, time_to_next = self._get_window(start_sec, window_size, merge_df, respevt_df, downsample=self.downsample_val, num_augment=num_augment)
                        if isinstance(window_df,  list):
                            num_original_1s += 1
                            for df in window_df:
                                window_tensor = torch.tensor(df.values)
                                X_list_1s.append(window_tensor)
                                y_list_1s.append(1)
                        else:
                            window_tensor = torch.tensor(window_df.values)
                            if not window_df.empty:
                                if time_to_next <= self.time_to_next_cut:
                                    X_list_1s.append(window_tensor)
                                    y_list_1s.append(1)
                                else:
                                    X_list_0s.append(window_tensor)
                                    y_list_0s.append(0)
                        start_sec += sliding_step_size
                
                sampled_X_list_0s = []
                sampled_y_list_0s = []
                print(f"  counts: 0s: {len(y_list_0s)}, 1s: {num_original_1s}, 1s+aug: {len(y_list_1s)}")
                sampled_idxs = random.sample(range(len(y_list_0s)), min([int(len(y_list_1s) * imbalance_factor), len(y_list_0s)]))
                for idx in sampled_idxs:
                    sampled_X_list_0s.append(X_list_0s[idx])
                    sampled_y_list_0s.append(y_list_0s[idx])

                # combine and shuffle
                X_list = X_list_1s + sampled_X_list_0s
                y_list = y_list_1s + sampled_y_list_0s
                zipped = list(zip(X_list, y_list))
                random.shuffle(zipped)
                X_list, y_list = zip(*zipped)
                X_list, y_list = list(X_list), list(y_list)

                self.data = torch.stack(X_list, 0)
                self.labels = torch.tensor(y_list, dtype=torch.float32)
                self.save_processed_data(cache_path)
                print(f"Processed data saved to {cache_path}")

        print(f"data __init__ unique labels = {self.labels.unique(return_counts=True)}")
        end = time.time()
        total_time_min = round((end - start) / 60, 2)
        print(f"  Finished __init__ in {total_time_min} mins")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx]
    
    def save_processed_data(self, save_path):
        torch.save({'data': self.data, 'labels': self.labels}, save_path)

    def load_processed_data(self, load_path):
        loaded_data = torch.load(load_path)
        self.data = loaded_data['data']
        self.labels = loaded_data['labels']
    
    def _load_data_and_merge(self, patient_id, permute_feature=None, perm_seed=None, drop_cols=None):
        respevt_df = self._read_respevt_txt_to_df(f"files/ucddb{patient_id:03}_respevt.txt")
        respevt_df['Time'] = respevt_df['Time'].dt.total_seconds()
        respevt_df = respevt_df.sort_values("Time")

        path_id = self._get_path_id(patient_id)
        edf_rec_df = pd.read_parquet(self._get_path(path_id))

        stage_df = pd.read_table(f"files/ucddb{patient_id:03}_stage.txt")
        stage_df['Time'] = (stage_df.index * SLEEP_EPOCH_S).astype(float)
        stage_df.rename(columns = {'0':'sleep_stage'}, inplace=True)

        event_timestamps = []
        for i in range(respevt_df.shape[0]):
            start_event_time = respevt_df.iloc[i, 0]
            duration = respevt_df.iloc[i, :]['Duration']
            for j in range(duration):
                event_timestamps.append(start_event_time + j)

        is_event = np.zeros(int(event_timestamps[-1] + 2))
        for time in event_timestamps:
            is_event[int(time)] = 1
        event_df = pd.DataFrame({
            'Time': range(len(is_event)),
            'is_event': is_event
        })
        event_df['Time'] = event_df['Time'].astype(float)

        merge_df = pd.merge_asof(edf_rec_df, stage_df, on='Time', direction='nearest')
        merge_df = pd.merge_asof(merge_df, event_df, on='Time', direction='nearest')
        merge_df = merge_df.astype('float32')
        merge_df = merge_df.set_index("Time")
        merge_df = merge_df.drop(['Right leg','Left leg'], axis=1, errors='ignore')

        if permute_feature is not None:
            assert perm_seed is not None, "column to permute provided, but no seed"
            np.random.seed(perm_seed)
            merge_df.iloc[:, permute_feature] = np.random.permutation(merge_df.iloc[:, permute_feature])

        if drop_cols is not None:
            merge_df = merge_df.drop(merge_df.columns[drop_cols], axis=1)
        
        return merge_df, respevt_df 
    
    def _trim_merge_df(self, merge_df):
        nonzero_idx = np.nonzero(merge_df['sleep_stage'])[0]
        first_idx = nonzero_idx[0]
        last_idx = nonzero_idx[-1]
        out_df = merge_df.iloc[first_idx:last_idx, :]
        return out_df
    
    def _get_window(self, start_sec, step_sec, merge_df, respevt_df, contain_event=False, downsample=None, num_augment=0):
        end_sec = start_sec + step_sec
        window_df = merge_df.query(f"index >= {start_sec} and index < {end_sec}")
        window_respevt_df = respevt_df.query(f"Time >= {start_sec} and Time < {end_sec}")

        max_apnea_secs = 16
        sampling_freq = 128
        max_apnea_count = max_apnea_secs * sampling_freq
        if (window_df['is_event'].sum() > max_apnea_count) and (contain_event == False):
            return pd.DataFrame(), pd.DataFrame(), np.nan
        
        window_df = window_df.drop(columns=['is_event'])
        
        next_respevt = respevt_df.query(f"Time >= {end_sec}")
        if next_respevt.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame(), np.nan

        next_event_time = next_respevt.iloc[0,:]['Time']
        time_to_next = next_event_time - end_sec

        if downsample is not None:
            total_rows = len(window_df)
            indices = np.linspace(0, total_rows - 1 - num_augment, downsample, dtype=int)
            if num_augment > 0 and time_to_next < self.time_to_next_cut:
                sampled_windows = []
                for i in range(num_augment):
                    indices_prime = indices + i
                    sampled_window = window_df.iloc[indices_prime]
                    sampled_windows.append(sampled_window)
                return sampled_windows, window_respevt_df, time_to_next

            window_df = window_df.iloc[indices]

        return window_df, window_respevt_df, time_to_next
    
    def _get_path(self, path_id):
        return f"{DATA_DIR}/patient_{path_id}.gzip"
    
    def _get_path_id(self, patient_id):
            return SORTED_IDS.index(patient_id)

    def _read_respevt_txt_to_df(self, filename):
        respevt_time = []
        respevt_type = []
        respevt_pb_cs = []
        respevt_duration = []
        respevt_desat_low = []
        respevt_desat_drop = []
        respevt_snore = []
        respevt_arousal = []
        respevt_bt_rate = []
        respevt_bt_change = []
        for line_no, line in enumerate(open(filename).readlines()):
            if line_no > 2:
                vals = line.split()
                is_pb = False
                is_no_desaturation = False
                for count, val in enumerate(vals):
                    if count == 0 and val != '\x1a':
                        respevt_time.append(val)
                    elif count == 1 and val == 'PB':
                        is_pb = True
                        respevt_type.append(val)
                    elif count == 1 and val != 'PB':
                        respevt_type.append(val)
                    elif count == 2 and is_pb:
                        pass
                    elif count == 2 and not is_pb:
                        respevt_pb_cs.append(np.nan)
                        respevt_duration.append(val)
                    elif count == 3 and is_pb:
                        respevt_pb_cs.append(val)
                    elif count == 3 and not is_pb:
                        if val == '-' or val == '+':
                            respevt_desat_low.append(np.nan)
                            respevt_desat_drop.append(np.nan)
                            respevt_snore.append(val)
                            is_no_desaturation = True
                        else:
                            respevt_desat_low.append(val)
                    elif count == 4 and is_pb:
                        respevt_duration.append(val)
                    elif count == 4 and not is_pb:
                        if is_no_desaturation:
                            respevt_arousal.append(val)
                        else:
                            respevt_desat_drop.append(val)
                    elif count == 5 and is_pb:
                        if val == '-' or val == '+':
                            respevt_desat_low.append(np.nan)
                            respevt_desat_drop.append(np.nan)
                            respevt_snore.append(val)
                            is_no_desaturation = True
                        else:
                            respevt_desat_low.append(val)
                    elif count == 5 and not is_pb:
                        if is_no_desaturation:
                            respevt_bt_rate.append(val)
                        else:
                            respevt_snore.append(val)
                    elif count == 6 and is_pb:
                        if is_no_desaturation:
                            respevt_arousal.append(val)
                        else:
                            respevt_desat_drop.append(val)
                    elif count == 6 and not is_pb:
                        if is_no_desaturation:
                            respevt_bt_change.append(val)
                        else:
                            respevt_arousal.append(val)
                    elif count == 7 and is_pb:
                        if is_no_desaturation:
                            respevt_bt_rate.append(val)
                        else:
                            respevt_snore.append(val)
                    elif count == 7 and not is_pb:
                        respevt_bt_rate.append(val)
                    elif count == 8 and is_pb:
                        if is_no_desaturation:
                            respevt_bt_change.append(val)
                        else:
                            respevt_arousal.append(val)
                    elif count == 8 and not is_pb:
                        respevt_bt_change.append(val)
                    elif count == 9 and is_pb:
                        respevt_bt_rate.append(val)
                    elif count == 10 and is_pb:
                        respevt_bt_change.append(val)
                if len(respevt_bt_rate) < len(respevt_time):
                    respevt_bt_rate.append(np.nan)
                if len(respevt_bt_change) < len(respevt_time):
                    respevt_bt_change.append(np.nan)
        df = pd.DataFrame(
            {
                'Time': respevt_time,
                'Type': respevt_type,
                'PB/CS': respevt_pb_cs,
                'Duration': respevt_duration,
                'DesaturationLow': respevt_desat_low,
                'Desaturation%Drop': respevt_desat_drop,
                'Snore': respevt_snore,
                'Arousal': respevt_arousal,
                'B_T_Rate': respevt_bt_rate,
                'B_T_Change': respevt_bt_change
            }
        )
        df['Time'] = pd.to_timedelta(df['Time'])
        df['Duration'] = pd.to_numeric(df['Duration'])
        df['DesaturationLow'] = pd.to_numeric(df['DesaturationLow'])
        df['Desaturation%Drop'] = pd.to_numeric(df['Desaturation%Drop'])
        df['B_T_Rate'] = pd.to_numeric(df['B_T_Rate'])
        df['B_T_Change'] = pd.to_numeric(df['B_T_Change'])
        return df


class SleepDataModule(L.LightningDataModule):
    def __init__(self, window_size, sliding_step_size, batch_size=16, patient_ids=SORTED_IDS, seed=42, downsample=None, time_to_next_cut=45, imbalance_factor=1, num_augment=0):
        super().__init__()
        self.window_size = window_size
        self.sliding_step_size = sliding_step_size
        self.patient_ids = patient_ids
        self.seed = seed
        self.n_patients = 25
        self.n_train = 17
        self.n_val = 3
        self.n_test = 5
        self.patient_order = random.Random(self.seed).sample(range(self.n_patients), self.n_patients)
        self.n_signals = 19
        self.downsample_val = downsample
        self.batch_size = batch_size
        self.time_to_next_cut = time_to_next_cut
        self.imbalance_factor = imbalance_factor
        self.num_augment = num_augment
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == 'fit':
            train_ids = []
            val_ids = []
            for i in range(self.n_train + self.n_val):
                patient_idx = self.patient_order[i]
                patient_id = self.patient_ids[patient_idx]
                if i < self.n_train:
                    train_ids.append(patient_id)
                else:
                    val_ids.append(patient_id)
            train_cache_path = f"data_cache/train_reduc_{'_'.join(map(str, train_ids))}_{self.window_size}_{self.sliding_step_size}_{self.downsample_val}_{self.time_to_next_cut}_{self.batch_size}_{self.imbalance_factor}_{self.num_augment}"
            val_cache_path = f"data_cache/val_reduc_{'_'.join(map(str, val_ids))}_{self.window_size}_{self.sliding_step_size}_{self.downsample_val}_{self.time_to_next_cut}_{self.batch_size}_{self.imbalance_factor}_{self.num_augment}"
            self.train_data = SleepDataSet(train_ids, self.window_size, self.sliding_step_size, downsample=self.downsample_val, time_to_next_cut=self.time_to_next_cut, cache_path=train_cache_path, num_augment=self.num_augment)
            self.val_data = SleepDataSet(val_ids, self.window_size, self.sliding_step_size, downsample=self.downsample_val, time_to_next_cut=self.time_to_next_cut, cache_path=val_cache_path, num_augment=self.num_augment)

        if stage == 'test' or stage == 'predict':
            test_ids = []
            start_test_idx = self.n_train + self.n_val
            for i in range(start_test_idx, start_test_idx + self.n_test):
                patient_idx = self.patient_order[i]
                patient_id = self.patient_ids[patient_idx]
                test_ids.append(patient_id)
            test_cache_path = f"data_cache/test_reduc_{'_'.join(map(str, test_ids))}_{self.window_size}_{self.sliding_step_size}_{self.downsample_val}_{self.time_to_next_cut}_{self.batch_size}_{self.imbalance_factor}_{self.num_augment}"
            self.test_data = SleepDataSet(test_ids, self.window_size, self.sliding_step_size, downsample=self.downsample_val, time_to_next_cut=self.time_to_next_cut, cache_path=test_cache_path, num_augment=self.num_augment)

    def train_dataloader(self):
        # 7.20 mins - num_workers=0, persistent_workers=False
        # 7.27 mins - num_workers=4, persistent_workers=False
        # needs more testing
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, persistent_workers=False) # true didn't seem to change performance
    
    def val_dataloader(self):
        # 1.17 mins - num_workers=0, persistent_workers=False
        # 1.24 mins - num_workers=4, persistent_workers=False
        # needs more testing
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0, persistent_workers=False) # true didn't seem to change performance
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=0)
    
    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=0)