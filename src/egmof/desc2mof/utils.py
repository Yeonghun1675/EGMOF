import torch
import pickle
import selfies
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
from . import __desc2mof_dir__
from .dataset import MOF_ENCODE_DICT, MOF_DECODE_DICT, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN, selfies2bb, CN_IDS


with open(f'{__desc2mof_dir__}/data/mof_topo_cn_dict.pkl', 'rb') as g:
    mof_topo_cn_dict =pickle.load(g)
topo_cn_dict = {MOF_ENCODE_DICT[key]: value for key, value in mof_topo_cn_dict.items()}

with open(f'{__desc2mof_dir__}/data/bb_cn_dict.pkl', 'rb') as g:
    bbname_cn_dict = pickle.load(g)
bb_cn_dict = {MOF_ENCODE_DICT[key]: value for key, value in bbname_cn_dict.items() if key in MOF_ENCODE_DICT}


with open(f'{__desc2mof_dir__}/data/cn.txt', 'r') as g:
    all_cn_list = g.read().strip().split()


def decode_token2mof(output, all_cn_list = all_cn_list):
    """
    Decodes token IDs back into MOF component strings.
    
    Args:
        output (Tensor or list): Tensor or list of token IDs [B, T].

    Returns:
        list: List of decoded MOF component lists, excluding EOS/PAD tokens.
    """
    mof_name_list = []
    
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().tolist()
    for tokens in (output):
        tkn_list =  [MOF_DECODE_DICT[token] for token in tokens if token not in [SEP_TOKEN, EOS_TOKEN, PAD_TOKEN]]
        tkn_list = ['+' if tkn in all_cn_list else tkn for tkn in tkn_list] 
        tkn_list = ''.join(tkn_list).split('+')
        new_tkn_list  = [selfies2bb[tkn] if tkn in selfies2bb else tkn for tkn in tkn_list  ]


        mof_name_list.append('+'.join(new_tkn_list))   
    return np.array(mof_name_list)



def is_valid(all_output, SEP_TOKEN_ID,
              topo_cn_dict = topo_cn_dict, bb_cn_dict = bb_cn_dict,
              selfies_cn_tkn = '[Lr]'
              ):
    """
    Args:
        all_output: List of model-generated tokens (List of List of Int)
        SEP_TOKEN_ID: ID of the [SEP] token that divides segments
    Returns:
        results: Boolean Array indicating validity
        log_list: Log list containing failure reasons
    """
    results = []
    log_list = []
    correct = 0
    LR_TOKEN = MOF_ENCODE_DICT[selfies_cn_tkn]  # Connection Point [Lr]
    
    # Helper function to split list by separator
    def split_list(lst, sep):
        grouped = []
        temp = []
        for val in lst:
            if val == sep:
                grouped.append(temp)
                temp = []
            else:
                temp.append(val)
        grouped.append(temp)
        return grouped

    for res in all_output:
        # 0. Basic preprocessing (remove EOS, PAD)
        seq = [token for token in res if token not in [EOS_TOKEN, PAD_TOKEN]]
        
        # Decoding name for logging (for debugging convenience)
        try:
            seq_name = [MOF_DECODE_DICT.get(tkn, str(tkn)) for tkn in seq]
        except:
            seq_name = str(seq)

        if not seq:
            results.append(False)
            log_list.append([seq_name, 'Not sequence'])
            continue

        # ---------------------------------------------------------
        # Rule 1: Check first token (Topology) and construct Expected CNs
        # ---------------------------------------------------------
        topo_id = seq[0]
        if topo_id not in topo_cn_dict:
            results.append(False)
            log_list.append([seq_name, f'Invalid Topology ID: {topo_id}'])
            continue
            
        # Construct ground truth (Expected CN List)
        # topo_cn_dict[id] -> (Node_CN_Array, Edge_Count)
        # Assume Edge CN is always 2
        node_cns_arr, edge_count = topo_cn_dict[topo_id]
        
        # Convert to int list as it might contain numpy arrays
        expected_cns = [int(x) for x in node_cns_arr] + [2] * int(edge_count)

        # ---------------------------------------------------------
        # Rule 2: Split by [SEP] and remove CN tokens
        # ---------------------------------------------------------
        body_seq = seq[1:] # Exclude Topology
        
        # 1. Primary split by SEP token
        raw_segments = split_list(body_seq, SEP_TOKEN_ID)
        
        # 2. Remove CN tokens inside each segment and filter empty segments
        clean_segments = []
        for seg in raw_segments:
            # Remove CN Token (Keep only pure BB tokens)
            seg_no_cn = [t for t in seg if t not in CN_IDS]
            
            # Consider as valid BB only if it contains content
            # (Remove empty lists caused by consecutive SEPs or trailing SEPs)
            if seg_no_cn:
                clean_segments.append(seg_no_cn)
        
        # ---------------------------------------------------------
        # Rule 3: Validate count and content
        # ---------------------------------------------------------
        # 3-1. Check if Segment (Building Block) count matches
        if len(clean_segments) != len(expected_cns):
            results.append(False)
            log_list.append([seq_name, f'Segment count mismatch. (Expected: {len(expected_cns)}, Got: {len(clean_segments)})'])
            continue
        
        # 3-2. Validate content of each segment
        segment_check_fail = False
        fail_reason = ""
        
        for idx, (segment, exp_cn) in enumerate(zip(clean_segments, expected_cns)):
            
            # [Case A] Single token -> Check bb_cn_dict
            if len(segment) == 1:
                token = segment[0]
                
                # If token is not in dictionary
                if token not in bb_cn_dict:
                    segment_check_fail = True
                    fail_reason = f"Unknown Single Token at idx {idx}: {token}"
                    break
                
                # Check if CN matches
                if bb_cn_dict[token] != exp_cn:
                    segment_check_fail = True
                    fail_reason = f"Single Token CN mismatch at idx {idx}. (Token CN: {bb_cn_dict[token]}, Expected: {exp_cn})"
                    break
            
            # [Case B] Multiple tokens -> Check [Lr] count
            else:
                actual_lr_count = segment.count(LR_TOKEN)
                
                if actual_lr_count != exp_cn:
                    segment_check_fail = True
                    fail_reason = f"Lr count mismatch at idx {idx}. (Count: {actual_lr_count}, Expected: {exp_cn})"
                    break
                
                try:
                    selfies_str =''.join([MOF_DECODE_DICT[seg] for seg in segment])
                    if selfies_str != selfies.encoder(selfies.decoder(selfies_str)):
                        segment_check_fail = True
                        fail_reason = f"New SELFIES but Invalid SELFIES at {idx}.)"    
                except Exception as e:
                    segment_check_fail = True
                    fail_reason = f"SELFIES ERROR AT {idx}: {e}"               

        if segment_check_fail:
            results.append(False)
            log_list.append([seq_name, fail_reason])
        else:
            results.append(True)
            correct += 1

    # Print results
    acc = correct / len(all_output) if len(all_output) > 0 else 0
    print(f'accuracy: {acc:.4f}, correct: {correct}, total_nums: {len(all_output)}')
    
    return np.array(results), log_list



def cal_wmse(pred_desc, target_desc, weights):
    mse = F.mse_loss(pred_desc, target_desc, reduction='none') # (N, D)
    w = torch.tensor(weights, dtype=mse.dtype, device=mse.device) # (D,)
    weighted_mse = mse * w # (N, D)
    wmse = weighted_mse.sum(dim=-1) / w.sum() # (N,)
    return wmse




