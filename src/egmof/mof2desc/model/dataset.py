import re
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from egmof.desc2mof import (
    PAD_TOKEN,
    SEP_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    MOF_ENCODE_DICT,
    MOF_DECODE_DICT,
    bb_cn_dict,
    bb2selfies,
    selfies2bb,
    __desc2mof_dir__,
)
from sklearn.preprocessing import StandardScaler


class Scaler:
    def __init__(self, mean, std, target_mean, target_std, eps: float = 1e-6):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps

    def to(self, dtype=None, device=None):
        self.mean = self.mean.to(dtype=dtype, device=device)
        self.std = self.std.to(dtype=dtype, device=device)

    def encode(self, batch):
        if isinstance(batch, torch.Tensor):
            return (batch - self.mean.to(batch.device)) / (self.std.to(batch.device) + self.eps)
        mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
        return (batch - mean) / (std + self.eps)

    def encode_target(self, target):
        return (target - self.target_mean) / (self.target_std + self.eps)

    def decode(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch * self.std.to(batch.device) + self.mean.to(batch.device)
        mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
        return batch * std + mean

    def decode_target(self, target):
        return target * self.target_std + self.target_mean


class CSVDataset(Dataset):
    def __init__(self, direc, scaled=True, scaler=None, feature_name_dir=None):
        super().__init__()
        self.direc = Path(direc).resolve()
        if not self.direc.exists():
            raise ValueError(f'{direc} does not exist.')
        self.data = pd.read_csv(direc)
        if feature_name_dir is None:
            feature_name_dir = f'{__desc2mof_dir__}/data/feature_name.txt'
        with open(feature_name_dir, 'r') as g:
            self.feature_names = [line.strip() for line in g.readlines()]
        self.encode_dict = MOF_ENCODE_DICT
        self.decode_dict = MOF_DECODE_DICT
        self.bb_cn_dict = bb_cn_dict
        self.bb2selfies = bb2selfies
        self.pad_token = PAD_TOKEN
        self.sep_token = SEP_TOKEN

        self.x = self.data[self.feature_names].to_numpy()
        split_list = [fname.split('.')[0].split('+') for fname in self.data['filename']]
        self.token_ids, self.attention_mask = make_target_data(
            split_list, self.encode_dict, self.pad_token,
        )

        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
        elif scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x = self.x.unsqueeze(-1)

    def __getitem__(self, idx):
        return [self.x[idx], self.token_ids[idx], self.attention_mask[idx]]

    def __len__(self):
        return len(self.x)


class MOF2DescGenDataset(Dataset):
    def __init__(self, direc):
        super().__init__()
        if isinstance(direc, str):
            direc_path = Path(direc).resolve()
            if not direc_path.exists():
                raise ValueError(f'{direc} does not exist.')
            self.data = pd.read_csv(direc_path)
        elif isinstance(direc, pd.DataFrame):
            self.data = direc.copy()
        else:
            raise TypeError("`direc` must be a filepath (str) or a pandas DataFrame.")
        self.data = self.data.dropna()
        self.encode_dict = MOF_ENCODE_DICT
        self.decode_dict = MOF_DECODE_DICT
        self.bb_cn_dict = bb_cn_dict
        self.bb2selfies = bb2selfies
        self.pad_token = PAD_TOKEN
        self.sep_token = SEP_TOKEN

        split_list = [fname.split('.')[0].split('+') for fname in self.data['filename']]
        self.token_ids, self.attention_mask = make_target_data(
            split_list, self.encode_dict, self.pad_token,
        )

    def __getitem__(self, idx):
        return [self.token_ids[idx], self.attention_mask[idx]]

    def __len__(self):
        return len(self.token_ids)


def make_target_data(split_list, ENCODE_DICT, PAD_TOKEN, max_len=512):
    token_ids_list, attention_list = [], []
    for splits in split_list:
        topo = splits[0]
        node_list = [item for item in splits if item.startswith('N')]
        edge_list = [item for item in splits if item.startswith('E')]

        target_name = [topo]
        for node in node_list:
            target_name += bb2tkn(node, ENCODE_DICT)
        for edge in edge_list:
            target_name += bb2tkn(edge, ENCODE_DICT)

        try:
            seq = [ENCODE_DICT[name] for name in target_name]
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Token '{missing}' not in encode_dict. Check bb2tkn() outputs / vocab building.")

        seq = seq[:max_len]
        attention = [1 for _ in range(len(seq))]
        if len(seq) < max_len:
            seq += [PAD_TOKEN] * (max_len - len(seq))
            attention += [0] * (max_len - len(attention))

        token_ids_list.append(seq)
        attention_list.append(attention)

    return torch.tensor(token_ids_list, dtype=torch.long), torch.tensor(attention_list, dtype=torch.long)


def bb2tkn(bb, encode_dict):
    cn = bb_cn_dict[bb]
    cn_tkn = [f'[CN_{cn}]']
    if bb in encode_dict:
        tkn = [bb]
    elif bb in bb2selfies:
        selfies_str = bb2selfies[bb]
        tkn = re.findall(r'\[[^\]]+\]', selfies_str)
    elif '[' in bb:
        selfies_str = bb
        tkn = re.findall(r'\[[^\]]+\]', selfies_str)
    else:
        raise KeyError(f"No corresponding SELFIES found for the edge: {bb}")
    return cn_tkn + tkn + ['[SEP]']


class Desc2MOFOutputDataset(Dataset):
    def __init__(self, generated_ids_list, x, scaled=False, scaler=None, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.sep_token = SEP_TOKEN

        self.x = x.numpy() if torch.is_tensor(x) else x

        cn_path = f'{__desc2mof_dir__}/data/cn.txt'
        with open(cn_path, 'r') as g:
            _ = g.read().strip().split()

        if torch.is_tensor(generated_ids_list):
            self.generated_ids_list = generated_ids_list.tolist()
        else:
            self.generated_ids_list = generated_ids_list

        self.clean_ids = [
            [token for token in t if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
            for t in self.generated_ids_list
        ]

        token_ids_list, attention_list = [], []
        for seq in self.clean_ids:
            seq = seq[:max_len]
            attention = [1 for _ in range(len(seq))]
            if len(seq) < max_len:
                seq += [PAD_TOKEN] * (max_len - len(seq))
                attention += [0] * (max_len - len(attention))
            token_ids_list.append(seq)
            attention_list.append(attention)

        self.token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_list, dtype=torch.long)

        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
        elif scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.x[idx], self.token_ids[idx], self.attention_mask[idx]


def bb2tkn(bb, encode_dict):
    cn = bb_cn_dict[bb]
    cn_tkn = [f'[CN_{cn}]']
    if bb in encode_dict:
        tkn = [bb]
    elif bb in bb2selfies:
        selfies_str = bb2selfies[bb]
        tkn = re.findall(r'\[[^\]]+\]', selfies_str)
    elif '[' in bb:
        selfies_str = bb
        tkn = re.findall(r'\[[^\]]+\]', selfies_str)
    else:
        raise KeyError(f"No corresponding SELFIES found for the edge: {bb}")
    return cn_tkn + tkn + ['[SEP]']


class Desc2MOFOutputDataset(Dataset):
    def __init__(self, generated_ids_list, x, scaled=False, scaler=None, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.sep_token = SEP_TOKEN

        self.x = x.numpy() if torch.is_tensor(x) else x

        cn_path = f'{__desc2mof_dir__}/data/cn.txt'
        with open(cn_path, 'r') as g:
            _ = g.read().strip().split()

        if torch.is_tensor(generated_ids_list):
            self.generated_ids_list = generated_ids_list.tolist()
        else:
            self.generated_ids_list = generated_ids_list

        self.clean_ids = [
            [token for token in t if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
            for t in self.generated_ids_list
        ]

        token_ids_list, attention_list = [], []
        for seq in self.clean_ids:
            seq = seq[:max_len]
            attention = [1 for _ in range(len(seq))]
            if len(seq) < max_len:
                seq += [PAD_TOKEN] * (max_len - len(seq))
                attention += [0] * (max_len - len(attention))
            token_ids_list.append(seq)
            attention_list.append(attention)

        self.token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_list, dtype=torch.long)

        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
        elif scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.x[idx], self.token_ids[idx], self.attention_mask[idx]
