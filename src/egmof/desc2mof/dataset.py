import re
import json
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from . import __desc2mof_dir__
from sklearn.preprocessing import StandardScaler
from collections import defaultdict




class CSVDataset(Dataset):
    """
    PyTorch Dataset for loading descriptor features (X) and creating 
    Transformer target sequences (Y) from a CSV file. 
    Applies standard scaling to the descriptor features.
    """
    def __init__(self, direc, max_len = 512, scaled = True, scaler = None, feature_name_dir =f'{__desc2mof_dir__}/../data/feature_name.txt'):
        """
        Args:
            direc (str): Path to the CSV file containing data.
            scaled (bool, optional): Whether to scale descriptor features (default: True).
            scaler (Scaler, optional): Pre-fit Scaler object for encoding (default: None).
            feature_name_dir (str): Path to the file listing feature names.
            max_len (int): topo (1) + node1 (1) + node2 (1) + [SEP] (1) + edge 1 (126) + [SEP] (1) + edge 2 (126) + (sos or eos (1))
        """
        super().__init__()

        # import CSV file
        self.direc = Path(direc).resolve()
        if not self.direc.exists():
            raise ValueError(f'{direc} does not exists.')
        self.data = pd.read_csv(direc)

        with open(feature_name_dir, 'r') as g:
            self.feature_names = [line.strip() for line in g.readlines()]
            
        self.max_len = max_len

        # make one dictionary for topology, node, edge class 
        self.encode_dict = MOF_ENCODE_DICT 
        self.sos_token = SOS_TOKEN  
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.sep_token = SEP_TOKEN
        self.decode_dict = MOF_DECODE_DICT

        


        self.x = self.data[self.feature_names].to_numpy()  #(self.data.iloc[:, :-1].to_numpy())
        self.input_seq, self.output_seq = self.make_target_data(self.max_len) 

        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
            

        if scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        # dimension: (D) -> (D,1)
        self.x = self.x.unsqueeze(-1) 

    def __getitem__(self, idx):
        return self.x[idx], self.input_seq[idx], self.output_seq[idx]
    
    def __len__(self):
        return len(self.x)

    def make_target_data(self, max_len):
        """
        Converts MOF filenames into two target sequences: 
        1. Input: [<SOS> + Target Tokens + <PAD>...] (shifted right)
        2. Output: [Target Tokens + <EOS> + <PAD>...] (actual target)
        Target Tokens:
                    ([SOS] (1)) + topo (1) 
            + cn (1) + node1 (or SELFIES) + [SEP] (1)
            + cn (1) + node2 (or SELFIES) + [SEP] (1) 
            + cn (1) + edge1 (or SELFIES) + [SEP] (1) 
            + cn (1) + edge2 (or SELFIES) + [SEP] (1) + ([EOS] (1)) 
        
        Args:
            max_len (int): Maximum sequence length for padding.

        Returns:
            tuple: (input_sequences [N, T], output_sequences [N, T])
        """
        input_seqs, output_seqs = [], []
        
        split_list = [fname.split('.')[0].split('+') for fname in self.data['filename']]
        for splits in split_list:
            topo = splits[0]
            node_list = [item for item in splits if item.startswith('N')]
            edge_list = [item for item in splits if item.startswith('E')]




            target_name = [topo]
            for node in node_list:
                target_name += bb2tkn(node, self.encode_dict,)

            for edge in edge_list:
                target_name += bb2tkn(edge,self.encode_dict,)
                
                    
            try:
                seq = [self.encode_dict[name] for name in target_name]
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(f"Token '{missing}' is not in encode_dict. Check bb2tkn() outputs / vocab building.")
           
        
            # Input: <sos> + seq, then pad/truncate
            inp = [self.sos_token] + seq
            inp = inp[:max_len]
            if len(inp) < max_len:
                inp += [self.pad_token] * (max_len - len(inp))            

            
            # Output: seq + <eos>, then pad/truncate
            out = seq + [self.eos_token]
            out = out[:max_len]
            if len(out) < max_len:
                out += [self.pad_token] * (max_len - len(out))
            
            input_seqs.append(inp)
            output_seqs.append(out)
            
        return torch.tensor(input_seqs, dtype=torch.long), torch.tensor(output_seqs, dtype=torch.long)

        
    def decode(self,value):
        """Decodes a single token ID back to its string name."""
        return self.decode_dict[value]
    





   



class MOFGenDataset(Dataset):
    """
    PyTorch Dataset used primarily for inference/prediction. 
    It only loads and preprocesses descriptor features (X), 
    as target sequences (Y) are not needed.
    """
    def __init__(self, direc, max_len = 512, scaled = True, scaler = None, feature_name_dir =f'{__desc2mof_dir__}/../data/feature_name.txt'):
        """
        Args:
            direc (str): Path to the CSV file containing data.
            scaled (bool, optional): Whether to scale descriptor features (default: True).
            scaler (Scaler, optional): Pre-fit Scaler object for encoding (default: None).
            feature_name_dir (str): Path to the file listing feature names.
            max_len (int): 
            ([SOS] (1)) + topo (1) 
            + cn (1) + node1 (or SELFIES) + [SEP] (1)
            + cn (1) + node2 (or SELFIES) + [SEP] (1) 
            + cn (1) + edge1 (or SELFIES) + [SEP] (1) 
            + cn (1) + edge2 (or SELFIES) + [SEP] (1) + ([EOS] (1))
        """
        super().__init__()

        # import CSV file
        if isinstance(direc, str):
            direc_path = Path(direc).resolve()
            if not direc_path.exists():
                raise ValueError(f'{direc} does not exist.')
            self.data = pd.read_csv(direc_path)
            
        elif isinstance(direc, pd.DataFrame):
            self.data = direc.copy()
        else:
            raise TypeError("`direc` must be a filepath (str) or a pandas DataFrame.")

        with open(feature_name_dir, 'r') as g:
            self.feature_names = [line.strip() for line in g.readlines()]
            
        self.max_len = max_len

        # make one dictionary for topology, node, edge class 
        self.encode_dict = MOF_ENCODE_DICT 
        self.sos_token = SOS_TOKEN  
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.sep_token = SEP_TOKEN
        self.decode_dict = MOF_DECODE_DICT
        

        self.x = self.data[self.feature_names].to_numpy()  
        self.x_origin = self.x.copy()

        
        if scaled and scaler is None:
            standard_scaler = StandardScaler()
            self.x = standard_scaler.fit_transform(self.x)
            mean = standard_scaler.mean_
            std = np.sqrt(standard_scaler.var_)
            self.scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)
            

        if scaled and scaler is not None:
            self.scaler = scaler
            self.x = self.scaler.encode(self.x)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        # dimension: (D) -> (D,1)
        self.x = self.x.unsqueeze(-1) 

    def __getitem__(self, idx):
        x = self.x[idx]
        x_origin = self.x_origin[idx]
        return x , x_origin
    
    def __len__(self):
        return len(self.x)


    
    def decode(self,value):
        return self.decode_dict[value]


def bb2tkn(bb, encode_dict):
    cn = bb_cn_dict[bb]
    cn_tkn = [f'[CN_{cn}]']

    if bb in encode_dict:
        tkn = [bb]
    elif bb in bb2selfies:
        selfies_str = bb2selfies[bb]
        tkn = re.findall(r'\[[^\]]+\]', selfies_str)
    else:
        raise KeyError(f"No corresponding SELFIES found for the edge: {bb}")
    return cn_tkn + tkn + ['[SEP]']



def make_encode_dict(file_name,PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN, T = 4):
    """
    Creates a mapping dictionary from component name (string) to token ID (integer). 
    The ID is determined by the line index in the input file(s).

    Args:
        file_name (str or list): Path(s) to the file(s) containing component names (one per line).
        T: special token ([PAD], [SOS], [EOS], [SEP])

    Returns:
        dict: Mapping of {component_name: token_ID}.
    """
    if isinstance(file_name, str):
        with open(file_name, ) as f:
            lines = f.readlines()

    elif isinstance(file_name, list):
        lines = []
        for name in file_name:
            with open(name, ) as f:
                lines += f.readlines()            
    
    class_dict = {name.strip(): i+T for i, name in enumerate(lines)}
    class_dict['[PAD]'] = PAD_TOKEN
    class_dict['[SOS]'] = SOS_TOKEN
    class_dict['[EOS]'] = EOS_TOKEN
    class_dict['[SEP]'] = SEP_TOKEN
    return class_dict
        
def make_decode_dict(topo_node_edge_dict, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN):
    """
    Creates the reverse mapping dictionary from token ID (integer) to string name.
    Includes special tokens: [SOS], [EOS], [PAD].

    Args:
        topo_node_edge_dict (dict): The original name-to-ID mapping.
        SOS_TOKEN (int): ID for Start-of-Sequence.
        EOS_TOKEN (int): ID for End-of-Sequence.
        PAD_TOKEN (int): ID for Padding.

    Returns:
        dict: Mapping of {token_ID: component_name_string}.
    """
    # Reverse the name-to-ID map
    decode_dict = {value: key for key, value in topo_node_edge_dict.items()}
    # Add special tokens

    decode_dict[PAD_TOKEN] = '[PAD]'
    decode_dict[SOS_TOKEN] = '[SOS]'
    decode_dict[EOS_TOKEN] = '[EOS]'
    decode_dict[SEP_TOKEN] = '[SEP]'
    
    return decode_dict


class Scaler(object):
    """
    A utility class for feature scaling (standardization), handling numpy arrays and PyTorch tensors.
    """
    def __init__(self, mean, std, target_mean, target_std, eps: float = 1e-6):
        """
        Args:
            mean (ndarray): Mean of the features.
            std (ndarray): Standard deviation of the features.
            target_mean (float): Mean of the target variable (used for decoding).
            target_std (float): Standard deviation of the target variable (used for decoding).
            eps (float): Small value for numerical stability (default: 1e-6).
        """
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps

    def to(self, dtype=None, device=None):
        self.mean = self.mean.to(dtype=dtype, device=device)
        self.std = self.std.to(dtype=dtype, device=device)

    def encode(self, batch):
        if isinstance(batch, torch.Tensor):
            return (batch - self.mean) / (self.std + self.eps)
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            return (batch - mean) / (std + self.eps)
    
    def encode_target(self, target):
        return (target - self.target_mean) / (self.target_std + self.eps)

    def decode(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch * self.std + self.mean
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            return batch * std + mean
    
    def decode_target(self, target):
        
        return target * self.target_std + self.target_mean


# --- Global Token and Dictionary Initialization ---

# 0. Load SELFIES ~ edge dict.
with open(f'{__desc2mof_dir__}/data/bb2selfies.pkl', 'rb') as g:
    bb2selfies = pickle.load(g)

with open(f'{__desc2mof_dir__}/data/selfies2bb.pkl', 'rb') as g:
    selfies2bb = pickle.load(g)

with open(f'{__desc2mof_dir__}/data/bb_cn_dict.pkl', 'rb') as g:
    bb_cn_dict = pickle.load(g)



# 1. Define component files
topo_node_edge_list = [f'{__desc2mof_dir__}/data/topology.txt', 
                       f'{__desc2mof_dir__}/data/metal_node.txt',
                       f'{__desc2mof_dir__}/data/metal_edge.txt',
                         f'{__desc2mof_dir__}/data/selfies.txt',
                         f'{__desc2mof_dir__}/data/cn.txt',
                         ]

# 2. Define special token IDs (IDs are assigned sequentially after component IDs)
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3

# 3. MOF NAME -> TOKEN NUM (Name-to-ID mapping)
MOF_ENCODE_DICT = make_encode_dict(topo_node_edge_list,PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN,)  

# 4. TOKEN NUM -> MOF NAME (ID-to-Name mapping)
MOF_DECODE_DICT = make_decode_dict(MOF_ENCODE_DICT, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN)

# 5. CN -> available tokens 

with open(f'{__desc2mof_dir__}/data/cn.txt', 'r') as g:
    cn_list = g.read().strip().split()

# CN_MBB_NAME_DICT, CN_MBB_NUM_DICT = defaultdict(list), defaultdict(list)
# for key, value in bb_cn_dict.items():
#     CN_MBB_NAME_DICT[value].append(key)
#     if key in MOF_ENCODE_DICT:
#         cn_tkn = [ cn for cn in cn_list if cn.endswith(f'_CN_{value}]')]
#         for cn in cn_tkn:
#             CN_MBB_NUM_DICT[MOF_ENCODE_DICT[cn]].append(MOF_ENCODE_DICT[key])

# CN_OBB_NUM_DICT= {}
# for cn in cn_list:
#     cn_enc = MOF_ENCODE_DICT[cn]
#     cn_num = int(cn.split('_')[-1][:-1])
#     CN_OBB_NUM_DICT[cn_enc] = cn_num


# # 6. MOF_TOPO_CN_DICT  e.g. 'acs' = ([[1878, 1889]], [[1874, 1885]]) 'hst'= ([[1876, 1887], [1875, 1886]], [[1874, 1885], [1874, 1885]])
# with open(f'{__desc2mof_dir__}//data/mof_topo_cn_dict.pkl', 'rb') as g:
#     mof_topo_cn_dict = pickle.load(g)

# edge_cn_name = ['[M_CN_2]','[O_CN_2]']
# edge_cn_encode = [MOF_ENCODE_DICT[tkn] for tkn in edge_cn_name]

# MOF_TOPO_CN_DICT = {}
# for topo_name, value in (mof_topo_cn_dict.items()):
#     topo_num = MOF_ENCODE_DICT[topo_name]
#     unique_cn, n_edge  = value
#     node_cn_list, edge_cn_list = [], []
#     for cn in unique_cn:
#         tmp = []
#         if f'[M_CN_{cn}]' in MOF_ENCODE_DICT:
#             tmp.append(MOF_ENCODE_DICT[f'[M_CN_{cn}]'])
#         if f'[O_CN_{cn}]' in MOF_ENCODE_DICT:
#             tmp.append(MOF_ENCODE_DICT[f'[O_CN_{cn}]'])
#         node_cn_list.append(tmp)

#     edge_cn_list = [edge_cn_encode]  if n_edge == 1 else [edge_cn_encode,edge_cn_encode ]

#     MOF_TOPO_CN_DICT[topo_num] = (node_cn_list, edge_cn_list)


# with open(f'{__desc2mof_dir__}/data/selfies.txt', 'r') as g:
#     all_selfies_list = g.read().strip().split()

# ALL_SELFIES_IDS = [MOF_ENCODE_DICT[c] for c in all_selfies_list]

# from collections import defaultdict
# bb_cn_reverse_dict = defaultdict(list)
# for bb , cn in bb_cn_dict.items():
#     if bb in MOF_ENCODE_DICT:
#         bb_cn_reverse_dict[cn].append(MOF_ENCODE_DICT[bb])


with open(f'{__desc2mof_dir__}/data/cn.txt', 'r') as g:
    CN_LIST = g.read().strip().split()

CN_IDS = [MOF_ENCODE_DICT[cn] for cn in CN_LIST]

# M_AVAILABLE_CN_BB_DICT , O_AVAILABLE_CN_BB_DICT  = {}, {}
# for cn_name in CN_LIST:
#     cn = int(cn_name.split('_')[-1][:1])

#     if cn_name.startswith('[M'):
        
#         M_AVAILABLE_CN_BB_DICT[MOF_ENCODE_DICT[cn_name]] = bb_cn_reverse_dict[cn]
#     else:
        
#         O_AVAILABLE_CN_BB_DICT[MOF_ENCODE_DICT[cn_name]] = cn


