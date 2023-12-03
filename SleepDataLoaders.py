import random
import pandas as pd
import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "rafael data download/merged"
SLEEP_EPOCH_S = 30
SORTED_IDS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

class SleepDataSet(Dataset):
    def __init__(self, patient_ids, window_size, sliding_step_size, n_signals):
        signal_length = window_size * 128 # 128Hz sampling freq
        
        X = torch.empty([0, n_signals, signal_length])
        y = torch.empty([0])

        for pid in patient_ids:
            merge_df, respevt_df = self._load_data_and_merge(pid)
            merge_df = self._trim_merge_df(merge_df)
            start_sec = merge_df.index[0]
            while (start_sec + window_size) < merge_df.index[-1]:
                window_df, _, time_to_next = self._get_window(start_sec, window_size, merge_df, respevt_df)
                if time_to_next != np.nan:
                    window_tensor = torch.tensor(window_df.values)
                    X = torch.cat((X, window_tensor), 0)
                    y = torch.cat((y, torch.tensor([time_to_next]))) 
                start_sec += sliding_step_size

        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx]
    
    def _load_data_and_merge(self, patient_id):
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
        merge_df = merge_df.set_index("Time")

        return merge_df, respevt_df 
    
    def _trim_merge_df(self, merge_df):
        nonzero_idx = np.nonzero(merge_df['sleep_stage'])[0]
        first_idx = nonzero_idx[0]
        last_idx = nonzero_idx[-1]
        out_df = merge_df.iloc[first_idx:last_idx, :]
        return out_df
    
    def _get_window(self, start_sec, step_sec, merge_df, respevt_df, contain_event=False):
        end_sec = start_sec + step_sec
        window_df = merge_df.query(f"index >= {start_sec} and index < {end_sec}")
        window_respevt_df = respevt_df.query(f"Time >= {start_sec} and Time < {end_sec}")

        if (1 in set(window_df['is_event'])) and (contain_event == False):
            return pd.DataFrame(), pd.DataFrame(), np.nan
        
        next_respevt = respevt_df.query(f"Time >= {end_sec}")
        if next_respevt.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame(), np.nan

        next_event_time = next_respevt.iloc[0,:]['Time']
        time_to_next = next_event_time - end_sec

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
    def __init__(self, window_size, sliding_step_size, patient_ids=SORTED_IDS, seed=42):
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
            
            self.train_data = SleepDataSet(train_ids, self.window_size, self.sliding_step_size, self.n_signals)
            self.val_data = SleepDataSet(val_ids, self.window_size, self.sliding_step_size, self.n_signals)

        if stage == 'test' or stage == 'predict':
            test_ids = []
            start_test_idx = self.n_train + self.n_val
            for i in range(start_test_idx, start_test_idx + self.n_test):
                patient_idx = self.patient_order[i]
                patient_id = self.patient_id[patient_idx]
                test_ids.append(patient_id)
            self.test_data = SleepDataSet(test_ids, self.window_size, self.sliding_step_size, self.n_signals)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32)
    
    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=32)