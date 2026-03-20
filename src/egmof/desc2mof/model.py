import os
import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from . import __desc2mof_dir__
from .utils import decode_token2mof, is_valid
from .dataset import MOF_ENCODE_DICT, MOF_DECODE_DICT, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN
 

class Desc2MOF(pl.LightningModule):
    """
    LightningModule for generating MOF structure tokens from a descriptor (Desc) 
    using a Transformer-based Encoder-Decoder model.
    
    Includes standard Seq2Seq training (Cross-Entropy loss) and optional 
    structural penalty loss (cal_combination_loss).
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # parameters
        self.hid_dim = config['hid_dim']
        self.num_heads = config.get('num_heads', 8)
        self.lr = config['learning_rate']
        self.warmup_step = config['warmup_steps']
        self.alpha = config.get('alpha', 0) # rmsd loss using locator btw topology and node
        self.max_len = config.get('max_len', 512)
        self.desc_dim = config['feature_size']
        self.topo_size = config['topo_size']
        self.cn_size  = config['cn_size']
        self.metal_node_size = config['metal_node_size'] 
        self.metal_edge_size = config['metal_edge_size']
        self.selfies_size = config['selfies_size']
        self.temperature = config.get('temperature', 1)
        self.max_token_len = config.get('max_token_len', 512)

            
        self.total_vocab_size = len(MOF_ENCODE_DICT)
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.sep_token = SEP_TOKEN


        # transformer encoder & decoder
        self.encoder = Encoder(self.config)
        self.model = Decoder(self.config,self.encoder,  self.hid_dim, self.total_vocab_size,
                             self.pad_token, self.sos_token, self.eos_token, self.sep_token,  
                             decoder_layers=config['num_layers'], num_heads = config['num_heads'], max_len = self.max_len )


        # # value net
        # self.value_net = nn.Sequential(
        #     nn.Flatten(),            # input: (B, 256, 1) → (B, 256)
        #     nn.Linear(self.hid_dim, self.hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hid_dim, 1)  # scalar value 
        # )        

                 
        self.validation_outputs = []
        self.test_outputs = []  
        self.pred_outputs = []


        special_size = 4

        self.topo_start = special_size
        self.topo_end   = self.topo_start + self.topo_size

        self.node_start = self.topo_end
        self.node_end   = self.node_start + self.metal_node_size          # Nxx “single tkns”

        self.edge_start = self.node_end
        self.edge_end   = self.edge_start + self.metal_edge_size          # Exx “single tkns”

        self.selfies_start = self.edge_end
        self.selfies_end   = self.selfies_start + self.selfies_size # SELFIES vocab size

        self.cn_start = self.selfies_end
        self.cn_end   = self.cn_start + self.cn_size                # CN vocab size (e.g: [CN_2]...[CN_8] etc)

        self.edge_cn_ids = torch.tensor([ MOF_ENCODE_DICT["[CN_2]"],], dtype=torch.long)
        self.node_cn_ids =  [MOF_ENCODE_DICT[f"[CN_{k}]"] for k in [3,4,5,6,7,8,9,10,12,24] ] 
        self.node_cn_ids  = torch.tensor(self.node_cn_ids, dtype=torch.long)



        assert self.cn_end == self.total_vocab_size            
        
    def forward(self,x,y):
        """
        Forward pass calling the underlying Decoder model.

        Args:
            x (Tensor): Source descriptor input.
            y (Tensor): Target token input (shifted right).

        Returns:
            Tensor: Logits [B, T, V].
        """       
        
        return self.model(x,y)

        
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step. Uses teacher forcing.

        Args:
            batch (list): [desc, target_input, target_output]
            desc: descriptor [B, D, 1]
            target_input: input tokens [B, 6] => [sos, topo, node1, node2, edge1, edge2]
            target_output: output tokens [B, 6] => [topo, node1, node2, edge1, edge2, eos]            
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Total loss (Cross-Entropy + alpha * Combi-Loss).
        """

        desc, target_input, target_output = batch
        logits = self(desc, target_input)  # [B, T, total_vocab_size]

        # 1. Cross-Entropy Loss (Seq2Seq)
        loss = F.cross_entropy(logits.view(-1, self.total_vocab_size), target_output.view(-1), ignore_index= self.pad_token)



            
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        
        return loss 
        
        

    def on_validation_start(self):
        self.validation_outputs = []
    
    def validation_step(self, batch, batch_idx):

        desc, target_input, target_output = batch
        logits = self(desc, target_input)

        # Loss (teacher forcing)
        loss = F.cross_entropy(
            logits.view(-1, self.total_vocab_size),
            target_output.view(-1),
            ignore_index=self.pad_token
        )


        # # Autoregressive accuracy
        # preds = self.generate(desc, )  # [B, T]
        # mask = target_output != self.pad_token                        # [B, T]
        # correct = (preds == target_output) & mask                     # [B, T]

        # # overall token accuracy
        # token_correct = correct.sum().item()
        # token_total = mask.sum().item()

        # # ---- class-wise accuracy (new vocab order) ----
        # # vocab: [PAD][SOS][EOS][SEP][TOPO...][NODE...][EDGE_SELFIES...]
        # special_size = 4
        # topo_start = special_size
        # topo_end = topo_start + self.topo_size

        # node_start = topo_end
        # node_end = node_start + self.node_size

        # # selfies start at node_end, ends at total_vocab_size
        # edge_start = node_end
        # edge_end = self.total_vocab_size

        # topo_mask = (target_output >= topo_start) & (target_output < topo_end) & mask
        # node_mask = (target_output >= node_start) & (target_output < node_end) & mask
        # edge_mask = (target_output >= edge_start) & (target_output < edge_end) & mask

        # topo_correct = (correct & topo_mask).sum().item()
        # node_correct = (correct & node_mask).sum().item()
        # edge_correct = (correct & edge_mask).sum().item()

        # topo_total = topo_mask.sum().item()
        # node_total = node_mask.sum().item()
        # edge_total = edge_mask.sum().item()

        # correct_pad = correct | ~mask                                   # treat pad positions as correct
        # all_correct = correct_pad.all(dim=1).float().sum().item()
        
        batch_size = desc.size(0)
        output = {
            "val_loss": loss.item() * batch_size,

            "batch_size": batch_size,
            # "token_correct": token_correct,
            # "token_total": token_total,
            # "all_correct": all_correct,
            
            # "topo_correct": topo_correct,
            # "topo_total": topo_total,
            # "node_correct": node_correct,
            # "node_total": node_total,
            # "edge_correct": edge_correct,
            # "edge_total": edge_total
        }
        self.validation_outputs.append(output)

        return output


        
    def on_validation_epoch_end(self):   

        totals = {k: sum(x[k] for x in self.validation_outputs) for k in [
            "val_loss","batch_size",
            # "token_correct", "token_total", "all_correct",
            # "topo_correct", "topo_total", "node_correct", "node_total",
            # "edge_correct", "edge_total", 
        ]}

        avg_loss = totals["val_loss"] / totals["batch_size"]

        # avg_token_acc = totals["token_correct"] / totals["token_total"] if totals["token_total"]>0 else 0.0
        # avg_all_correct = totals["all_correct"] / totals["batch_size"]
        # avg_topo_acc = totals["topo_correct"] / totals["topo_total"] if totals["topo_total"]>0 else 0.0
        # avg_node_acc = totals["node_correct"] / totals["node_total"] if totals["node_total"]>0 else 0.0
        # avg_edge_acc = totals["edge_correct"] / totals["edge_total"] if totals["edge_total"]>0 else 0.0

        self.log("val/avg_val_loss", avg_loss, prog_bar=True, sync_dist=True)

        # self.log("val/avg_token_acc", avg_token_acc, prog_bar=True, sync_dist=True)
        # self.log("val/avg_all_correct", avg_all_correct, prog_bar=True, sync_dist=True)

        # self.log("val/avg_topo_acc", avg_topo_acc, prog_bar=False, sync_dist=True)
        # self.log("val/avg_node_acc", avg_node_acc, prog_bar=False, sync_dist=True)
        # self.log("val/avg_edge_acc", avg_edge_acc, prog_bar=False, sync_dist=True)


        

    def on_test_start(self):
        self.test_outputs = []
        
    def test_step(self, batch, batch_idx):

        desc, target_input, target_output = batch

        #mask
        mask = target_output != self.pad_token                        # [B, T]
 
        token_total = mask.sum()
        
        logits = self(desc, target_input)

        # Loss (teacher forcing)
        loss = F.cross_entropy(
            logits.view(-1, self.total_vocab_size),
            target_output.view(-1),
            ignore_index=self.pad_token
        )



        # ---- class-wise masks (new vocab order) ----
        # vocab: [PAD][SOS][EOS][SEP][TOPO...][metal node..][metal edge]..[SELFIES...][CN...]


        topo_mask    = (target_output >= self.topo_start)    & (target_output < self.topo_end)    & mask
        node_mask    = (target_output >= self.node_start)    & (target_output < self.node_end)    & mask
        edge_mask    = (target_output >= self.edge_start)    & (target_output < self.edge_end)    & mask
        selfies_mask = (target_output >= self.selfies_start) & (target_output < self.selfies_end) & mask
        cn_mask      = (target_output >= self.cn_start)      & (target_output < self.cn_end)      & mask


        topo_total    = topo_mask.sum().item()
        node_total    = node_mask.sum().item()
        edge_total    = edge_mask.sum().item()
        selfies_total = selfies_mask.sum().item()
        cn_total      = cn_mask.sum().item()


        # Top1 accuracy (teacher-forcing)
        preds_top1 = logits.argmax(dim=-1)  # [B, T]
        correct_top1 = (preds_top1 == target_output) & mask

        token_correct_top1 = correct_top1.sum().item()
        topo_correct_top1 = (correct_top1 & topo_mask).sum().item()
        node_correct_top1 = (correct_top1 & node_mask).sum().item()
        edge_correct_top1 = (correct_top1 & edge_mask).sum().item()
        selfies_correct_top1 = (correct_top1 & selfies_mask).sum().item()
        cn_correct_top1 = (correct_top1 & cn_mask).sum().item()

        correct_pad_top1 = correct_top1 | ~mask
        all_correct_top1 = correct_pad_top1.all(dim=1).float().sum().item() # [B] -> scalar mean
        # block-level (node/edge segment all-correct)
        node_all_correct_top1, node_all_total_top1 = self._block_all_correct_total_from_preds(
            preds_top1, target_output, self.node_cn_ids
        )
        edge_all_correct_top1, edge_all_total_top1 = self._block_all_correct_total_from_preds(
            preds_top1, target_output, self.edge_cn_ids
        )       

        # --------------------------
        # Top5 accuracy (token-level + block-level)
        # --------------------------
        tok_acc_top5, seq_all_top5, correct_top5, mask_top5 = topk_token_metrics(
            logits, target_output, self.pad_token, k=5
        )
        # NOTE: topk_token_metrics may return a mask; use it for consistency
        # but in most implementations mask_top5 == mask
        token_correct_top5 = correct_top5.sum().item()

        topo_correct_top5 = (correct_top5 & topo_mask).sum().item()
        node_correct_top5 = (correct_top5 & node_mask).sum().item()
        edge_correct_top5 = (correct_top5 & edge_mask).sum().item()
        selfies_correct_top5 = (correct_top5 & selfies_mask).sum().item()
        cn_correct_top5 = (correct_top5 & cn_mask).sum().item()

        correct_pad_top5 = correct_top5 | ~mask
        all_correct_top5 = correct_pad_top5.all(dim=1).float().sum().item()

        node_all_correct_top5, node_all_total_top5 = self._block_all_correct_total_from_correct_matrix(
            correct_top5, target_output, self.node_cn_ids
        )
        edge_all_correct_top5, edge_all_total_top5 = self._block_all_correct_total_from_correct_matrix(
            correct_top5, target_output, self.edge_cn_ids
        )

        
        
        # Autoregressive accuracy
        preds = self.generate(desc, )  # [B, T] #max_token_len=target_output.size(1)
        correct = (preds == target_output) & mask                     # [B, T]

        token_correct = correct.sum().item()



        topo_correct = (correct & topo_mask).sum().item()
        node_correct = (correct & node_mask).sum().item()
        edge_correct = (correct & edge_mask).sum().item()
        selfies_correct = (correct & selfies_mask).sum().item()
        cn_correct = (correct & cn_mask).sum().item()

        correct_pad = correct | ~mask
        all_correct = correct_pad.all(dim=1).float().sum().item()

        node_all_correct, node_all_total = self._block_all_correct_total_from_preds(
            preds, target_output, self.node_cn_ids
        )
        edge_all_correct, edge_all_total = self._block_all_correct_total_from_preds(
            preds, target_output, self.edge_cn_ids
        )

        # totals should match across the three metrics (same target_output 기준)
        assert node_all_total == node_all_total_top1 == node_all_total_top5


        batch_size = desc.size(0)

        output = {
            "batch_size": batch_size,
            "test_loss": loss.item() * batch_size,


            # ---- autoregressive (generate) ----
            "token_correct": token_correct,
            "all_correct": all_correct,
            "topo_correct": topo_correct,
            "node_correct": node_correct,
            "edge_correct": edge_correct,
            "selfies_correct": selfies_correct,
            "cn_correct": cn_correct,
            "node_all_correct": node_all_correct,
            "edge_all_correct": edge_all_correct,

            # ---- top1 teacher-forcing ----
            "token_correct_top1": token_correct_top1,
            "all_correct_top1": all_correct_top1,
            "topo_correct_top1": topo_correct_top1,
            "node_correct_top1": node_correct_top1,
            "edge_correct_top1": edge_correct_top1,
            "selfies_correct_top1": selfies_correct_top1,
            "cn_correct_top1": cn_correct_top1,
            "node_all_correct_top1": node_all_correct_top1,
            "edge_all_correct_top1": edge_all_correct_top1,

            # ---- top5 teacher-forcing ----
            "token_correct_top5": token_correct_top5,
            "all_correct_top5": all_correct_top5,
            "topo_correct_top5": topo_correct_top5,
            "node_correct_top5": node_correct_top5,
            "edge_correct_top5": edge_correct_top5,
            "selfies_correct_top5": selfies_correct_top5,
            "cn_correct_top5": cn_correct_top5,
            "node_all_correct_top5": node_all_correct_top5,
            "edge_all_correct_top5": edge_all_correct_top5,

            # ---- totals (denominators) ----
            "token_total": token_total,
            "topo_total": topo_total,
            "node_total": node_total,
            "edge_total": edge_total,
            "selfies_total": selfies_total,
            "cn_total": cn_total,
            "node_all_total": node_all_total,
            "edge_all_total": edge_all_total,

            # optional debug
            "pred_tokens": preds,
            # "pred_tokens_top1": preds_top1,
        }

            
        self.test_outputs.append(output)

        return output

    def on_test_epoch_end(self):   

        # ---- aggregate sums ----
        keys = [
            "test_loss", "batch_size", "token_total",
            "topo_total", "node_total", "edge_total", "selfies_total", "cn_total",
            "node_all_total", "edge_all_total",   # block-level denom

            # autoregressive (generate)
            "token_correct","all_correct",
            "topo_correct", "node_correct", "edge_correct", "selfies_correct", "cn_correct",
            "node_all_correct", "edge_all_correct",

            # top1
            "token_correct_top1", "all_correct_top1",
            "topo_correct_top1", "node_correct_top1", "edge_correct_top1", "selfies_correct_top1", "cn_correct_top1",
            "node_all_correct_top1", "edge_all_correct_top1",

            # top5
            "token_correct_top5", "all_correct_top5",
            "topo_correct_top5", "node_correct_top5", "edge_correct_top5", "selfies_correct_top5", "cn_correct_top5",
            "node_all_correct_top5", "edge_all_correct_top5",
        ]

        totals = {k: sum(x[k] for x in self.test_outputs) for k in keys}

        # ---- averages / rates ----
        avg_loss = totals["test_loss"] / totals["batch_size"] if totals["batch_size"] > 0 else 0.0

        def safe_div(num, den):
            return (num / den) if den and den > 0 else 0.0

        # autoregressive (generate)
        avg_token_acc = safe_div(totals["token_correct"], totals["token_total"])
        avg_all_correct = safe_div(totals["all_correct"], totals["batch_size"])
        avg_topo_acc = safe_div(totals["topo_correct"], totals["topo_total"])
        avg_node_acc = safe_div(totals["node_correct"], totals["node_total"])
        avg_edge_acc = safe_div(totals["edge_correct"], totals["edge_total"])
        avg_selfies_acc = safe_div(totals["selfies_correct"], totals["selfies_total"])
        avg_cn_acc = safe_div(totals["cn_correct"], totals["cn_total"])
        avg_node_all_acc = safe_div(totals["node_all_correct"], totals["node_all_total"])
        avg_edge_all_acc = safe_div(totals["edge_all_correct"], totals["edge_all_total"])

        # top1
        avg_token_acc_top1 = safe_div(totals["token_correct_top1"], totals["token_total"])
        avg_all_correct_top1 = safe_div(totals["all_correct_top1"], totals["batch_size"])
        avg_topo_acc_top1 = safe_div(totals["topo_correct_top1"], totals["topo_total"])
        avg_node_acc_top1 = safe_div(totals["node_correct_top1"], totals["node_total"])
        avg_edge_acc_top1 = safe_div(totals["edge_correct_top1"], totals["edge_total"])
        avg_selfies_acc_top1 = safe_div(totals["selfies_correct_top1"], totals["selfies_total"])
        avg_cn_acc_top1 = safe_div(totals["cn_correct_top1"], totals["cn_total"])
        avg_node_all_acc_top1 = safe_div(totals["node_all_correct_top1"], totals["node_all_total"])
        avg_edge_all_acc_top1 = safe_div(totals["edge_all_correct_top1"], totals["edge_all_total"])

        # top5
        avg_token_acc_top5 = safe_div(totals["token_correct_top5"], totals["token_total"])
        avg_all_correct_top5 = safe_div(totals["all_correct_top5"], totals["batch_size"])
        avg_topo_acc_top5 = safe_div(totals["topo_correct_top5"], totals["topo_total"])
        avg_node_acc_top5 = safe_div(totals["node_correct_top5"], totals["node_total"])
        avg_edge_acc_top5 = safe_div(totals["edge_correct_top5"], totals["edge_total"])
        avg_selfies_acc_top5 = safe_div(totals["selfies_correct_top5"], totals["selfies_total"])
        avg_cn_acc_top5 = safe_div(totals["cn_correct_top5"], totals["cn_total"])
        avg_node_all_acc_top5 = safe_div(totals["node_all_correct_top5"], totals["node_all_total"])
        avg_edge_all_acc_top5 = safe_div(totals["edge_all_correct_top5"], totals["edge_all_total"])

        # ---- save predictions (optional) ----
        all_pred_tokens = torch.concat([x["pred_tokens"] for x in self.test_outputs], dim=0)
        mof_name_pred = ["+".join(token) for token in decode_token2mof(all_pred_tokens)]
        if self.config.get("save_test", False):
            exp_name = self.config["exp_name"]
            pd.DataFrame(mof_name_pred, columns=["filename"]).to_csv(f"{exp_name}_mof_name_pred.csv", index=None)

        # ---- logging ----
        self.log("test/avg_test_loss", avg_loss, prog_bar=True, sync_dist=True)

        # autoregressive (generate)
        self.log("test/avg_token_acc", avg_token_acc, prog_bar=True, sync_dist=True)
        self.log("test/avg_all_correct", avg_all_correct, prog_bar=True, sync_dist=True)
        self.log("test/avg_topo_acc", avg_topo_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_acc", avg_node_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_acc", avg_edge_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_selfies_acc", avg_selfies_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_cn_acc", avg_cn_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_all_acc", avg_node_all_acc, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_all_acc", avg_edge_all_acc, prog_bar=False, sync_dist=True)

        # top1
        self.log("test/avg_token_acc_top1", avg_token_acc_top1, prog_bar=True, sync_dist=True)
        self.log("test/avg_all_correct_top1", avg_all_correct_top1, prog_bar=True, sync_dist=True)
        self.log("test/avg_topo_acc_top1", avg_topo_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_acc_top1", avg_node_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_acc_top1", avg_edge_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_selfies_acc_top1", avg_selfies_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_cn_acc_top1", avg_cn_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_all_acc_top1", avg_node_all_acc_top1, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_all_acc_top1", avg_edge_all_acc_top1, prog_bar=False, sync_dist=True)

        # top5
        self.log("test/avg_token_acc_top5", avg_token_acc_top5, prog_bar=True, sync_dist=True)
        self.log("test/avg_all_correct_top5", avg_all_correct_top5, prog_bar=True, sync_dist=True)
        self.log("test/avg_topo_acc_top5", avg_topo_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_acc_top5", avg_node_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_acc_top5", avg_edge_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_selfies_acc_top5", avg_selfies_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_cn_acc_top5", avg_cn_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_node_all_acc_top5", avg_node_all_acc_top5, prog_bar=False, sync_dist=True)
        self.log("test/avg_edge_all_acc_top5", avg_edge_all_acc_top5, prog_bar=False, sync_dist=True)



    def on_predict_epoch_start(self):
        self.pred_outputs = []


    def predict_step(self, batch, batch_idx):
        desc, desc_origin = batch
        batch_size = desc.shape[0]

        # Autoregressive accuracy
        pred = self.generate(desc, temperature = self.temperature )  # [B, T]

        self.pred_outputs.append( {
            'desc': desc_origin.detach().cpu(),
            'pred': pred.detach().cpu()
        })



    def on_predict_epoch_end(self):  
        feature_name_dir = self.config.get('feature_name_dir',f'{__desc2mof_dir__}/../desc2mof/data/feature_name.txt')
        with open(feature_name_dir, 'r') as g:
            feature_names = [line.strip() for line in g.readlines()]
        exp_name = self.config.get('exp_name', 'pred')

        all_output = [ output['pred'] for output in self.pred_outputs]
        all_input = [ output['desc'] for output in self.pred_outputs]

        all_output = torch.concat(all_output)
        all_input = torch.concat(all_input)
        mof_names = decode_token2mof(all_output)

        valid_mask, log_list = is_valid(all_output, SEP_TOKEN_ID=SEP_TOKEN)

        final_preds = all_output[valid_mask]
        final_mof = mof_names[valid_mask].reshape(-1, 1)
        final_descs = all_input[valid_mask].reshape(-1, self.desc_dim )
        final_outputs = (np.concatenate([final_descs, final_mof], axis=1))
        final_df = pd.DataFrame(final_outputs, columns = feature_names+['filename'])
        final_df.to_csv(f"{exp_name}_valid_pred_mof.csv", index=None)

    def on_predict_epoch_end(self):  
        # 1. 현재 GPU에 쌓인 데이터들을 텐서로 합칩니다.
        local_desc = torch.cat([out['desc'] for out in self.pred_outputs], dim=0).to(self.device)
        local_pred = torch.cat([out['pred'] for out in self.pred_outputs], dim=0).to(self.device)

        # 2. 모든 GPU(Rank)로부터 데이터를 모읍니다 (all_gather)
        # self.all_gather를 쓰면 [Rank0_tensor, Rank1_tensor] 형태의 리스트가 반환됩니다.
        all_desc_tensors = self.all_gather(local_desc)
        all_pred_tensors = self.all_gather(local_pred)

        # 3. 메인 프로세스(Rank 0)에서만 파일 저장을 수행합니다.
        if self.trainer.is_global_zero:
            # all_gather 결과는 [Rank, Batch, ...] 형태이므로 차원을 합쳐줍니다.
            all_input = torch.cat(list(all_desc_tensors), dim=0).cpu()
            all_output = torch.cat(list(all_pred_tensors), dim=0).cpu()

            # --- 여기서부터는 기존 로직과 동일합니다 ---
            feature_name_dir = self.config.get('feature_name_dir', f'{__desc2mof_dir__}/../desc2mof/data/feature_name.txt')
            with open(feature_name_dir, 'r') as g:
                feature_names = [line.strip() for line in g.readlines()]
            
            exp_name = self.config.get('exp_name', 'pred')
            mof_names = decode_token2mof(all_output)
            valid_mask, log_list = is_valid(all_output, SEP_TOKEN_ID=SEP_TOKEN)

            final_preds = all_output[valid_mask]
            final_mof = mof_names[valid_mask].reshape(-1, 1)
            final_descs = all_input[valid_mask].reshape(-1, self.desc_dim)
            
            final_outputs = np.concatenate([final_descs, final_mof], axis=1)
            final_df = pd.DataFrame(final_outputs, columns=feature_names + ['filename'])
            final_df.to_csv(f"{exp_name}_valid_pred_mof.csv", index=None)
            
            # 메모리 비우기
            self.pred_outputs.clear()

    def _block_all_correct_total_from_preds(
        self,
        pred_ids: torch.Tensor,    # [B, T]
        tgt_ids: torch.Tensor,     # [B, T]
        cn_ids: torch.Tensor,      # [K]  (node CN들 or edge CN들)
    ):
        """
        블록 정의: CN 토큰 위치 s 에서 시작, 그 뒤 첫 [SEP] 위치 e 까지를 하나의 블록으로 본다.
                블록 구간은 [s, e] (SEP 포함)로 판정.
        반환: (all_correct_blocks, total_blocks)
        """
        device = tgt_ids.device
        B, T = tgt_ids.shape

        sep_id = self.sep_token
        pad_id = self.pad_token
        cn_ids = cn_ids.to(device)

        # target에서 topo 이후 PAD 전까지만 고려 (PAD가 없으면 T)
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
        is_pad = (tgt_ids == pad_id)
        has_pad = is_pad.any(dim=1)  # [B]
        first_pad = is_pad.float().argmax(dim=1)  # PAD 없으면 0 나올 수 있음
        first_pad = torch.where(has_pad, first_pad, torch.full_like(first_pad, T))

        valid_pos = idx < first_pad.unsqueeze(1)   # [B,T]
        is_sep = (tgt_ids == sep_id) & valid_pos

        # CN 위치 마스크 (torch.isin 대체: (B,T,K) 비교 후 any)
        is_cn = (tgt_ids.unsqueeze(-1) == cn_ids.view(1, 1, -1)).any(dim=-1) & valid_pos  # [B,T]

        # 최대 블록 개수는 작음: node<=2, edge<=2 라면 여기서 Kmax=2로 두면 됨
        Kmax = 2

        # 시작 인덱스 추출: CN이면 idx, 아니면 T(큰 값)로 해서 작은 것부터 K개
        start_candidates = torch.where(is_cn, idx, torch.full_like(idx, T))  # [B,T]
        starts, _ = torch.topk(start_candidates, k=Kmax, largest=False, dim=1)  # [B,Kmax]

        valid_block = starts < first_pad.unsqueeze(1)  # [B,Kmax]  (start==T면 invalid)

        # 각 start에 대해 "뒤에서 첫 SEP" 찾기: (B,K,T) 마스크 만든 뒤 argmax
        # after_start & is_sep
        after_start = idx.unsqueeze(1) > starts.unsqueeze(-1)  # [B,K,T] idx[B,1,512] starts[B,2,1] -> [B,2,512]
        sep_after = after_start & is_sep.unsqueeze(1)          # [B,K,T] [B,2,512]

        # sep가 없으면 argmax=0이 되므로, has_sep_after로 보정
        has_sep_after = sep_after.any(dim=-1)  # [B,K]
        ends = sep_after.float().argmax(dim=-1)  # [B,K]  (첫 True 위치)
        ends = torch.where(has_sep_after, ends, torch.full_like(ends, T))  # 없으면 T로

        # 블록 범위 마스크: start <= pos <= end (SEP 포함)
        pos = idx.unsqueeze(1)  # [B,1,T]
        block_mask = (pos >= starts.unsqueeze(-1)) & (pos <= ends.unsqueeze(-1))  # [B,K,T]
        block_mask = block_mask & valid_block.unsqueeze(-1)  # invalid 블록은 전부 False

        # 토큰 일치
        eq = (pred_ids == tgt_ids) & valid_pos  # [B,T]
        eq = eq.unsqueeze(1)  # [B,1,T]

        # 블록이 맞으려면, 블록 마스크 True인 위치가 전부 eq=True
        # => (eq | ~block_mask).all(T)
        block_ok = (eq | ~block_mask).all(dim=-1)  # [B,K]

        correct = (block_ok & valid_block).sum().item()
        total = valid_block.sum().item()
        return float(correct), float(total)

    def _block_all_correct_total_from_correct_matrix(
        self,
        correct_matrix: torch.Tensor,  # [B, T] bool
        tgt_ids: torch.Tensor,         # [B, T]
        cn_ids: torch.Tensor,          # [K]
    ):
        device = tgt_ids.device
        B, T = tgt_ids.shape

        sep_id = self.sep_token
        pad_id = self.pad_token
        cn_ids = cn_ids.to(device)

        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        is_pad = (tgt_ids == pad_id)
        has_pad = is_pad.any(dim=1)
        first_pad = is_pad.float().argmax(dim=1)
        first_pad = torch.where(has_pad, first_pad, torch.full_like(first_pad, T))

        valid_pos = idx < first_pad.unsqueeze(1)
        is_sep = (tgt_ids == sep_id) & valid_pos
        is_cn = (tgt_ids.unsqueeze(-1) == cn_ids.view(1, 1, -1)).any(dim=-1) & valid_pos

        Kmax = 2
        start_candidates = torch.where(is_cn, idx, torch.full_like(idx, T))
        starts, _ = torch.topk(start_candidates, k=Kmax, largest=False, dim=1)

        valid_block = starts < first_pad.unsqueeze(1)

        after_start = idx.unsqueeze(1) > starts.unsqueeze(-1)
        sep_after = after_start & is_sep.unsqueeze(1)
        has_sep_after = sep_after.any(dim=-1)
        ends = sep_after.float().argmax(dim=-1)
        ends = torch.where(has_sep_after, ends, torch.full_like(ends, T))

        pos = idx.unsqueeze(1)
        block_mask = (pos >= starts.unsqueeze(-1)) & (pos <= ends.unsqueeze(-1))
        block_mask = block_mask & valid_block.unsqueeze(-1)

        eq = correct_matrix & valid_pos  # [B,T] bool
        eq = eq.unsqueeze(1)

        block_ok = (eq | ~block_mask).all(dim=-1)

        correct = (block_ok & valid_block).sum().item()
        total = valid_block.sum().item()
        return float(correct), float(total)
   
    
    def generate(self, desc, temperature = 1):

        return self.model.generate(desc, pad_token = self.pad_token, eos_token=self.eos_token, max_token_len=self.max_token_len, temperature = temperature)


    # def forward_value(self, desc):
    #     B = desc.size(0)
    #     device = desc.device  
        
    #     x = self.model.pos_encoder(desc)
    #     x = self.model.encoder(x) # (B, T, H)
    #     x = x[:, 0, :]
    #     x = self.value_net(x)  # shape: (B, 1)    
    #     return x.squeeze(-1)

    
    def configure_optimizers(self, ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps

        if isinstance(self.warmup_step, float):
            warmup_steps = int(max_steps * self.warmup_step)
        else:
            warmup_steps = self.warmup_step


        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )

    def _edge_all_correct_total_from_preds(self, pred_ids, tgt_ids):
        """
        pred_ids, tgt_ids: [B, T]
        returns:
        edge_all_correct: float  (맞춘 edge 세그먼트 개수 = edge1_correct + edge2_correct)
        edge_all_total:   float  (존재하는 edge 세그먼트 개수 = edge1_total + edge2_total)
        """
        B, T = tgt_ids.shape
        edge_all_correct = 0.0
        edge_all_total = 0.0

        for b in range(B):
            tgt = tgt_ids[b]
            pred = pred_ids[b]

            sep_pos = (tgt == self.sep_token).nonzero(as_tuple=False).view(-1)
            eos_pos = (tgt == self.eos_token).nonzero(as_tuple=False).view(-1)
            if sep_pos.numel() < 2 or eos_pos.numel() < 1:
                continue

            sep1 = sep_pos[0].item()
            sep2 = sep_pos[1].item()
            eos  = eos_pos[0].item()

            # edge1: 항상 존재(비어있어도 edge1로 취급할지? 보통은 존재한다고 봄)
            edge_all_total += 1.0
            edge1_ok = (pred[sep1 + 1: sep2] == tgt[sep1 + 1: sep2]).all().item() if (sep2 > sep1 + 1) else True
            if edge1_ok:
                edge_all_correct += 1.0

            # edge2: span 길이가 0이면 "없다"로 간주 -> total에 포함 X
            has_edge2 = (eos > sep2 + 1)
            if has_edge2:
                edge_all_total += 1.0
                edge2_ok = (pred[sep2 + 1: eos] == tgt[sep2 + 1: eos]).all().item()
                if edge2_ok:
                    edge_all_correct += 1.0

        return edge_all_correct, edge_all_total

    def _edge_all_correct_total_from_correct_matrix(self, correct_matrix, tgt_ids):
        """
        correct_matrix: [B, T] bool
        tgt_ids: [B, T]
        returns: (edge_all_correct, edge_all_total)
        """
        B, T = tgt_ids.shape
        edge_all_correct = 0.0
        edge_all_total = 0.0

        for b in range(B):
            tgt = tgt_ids[b]
            cor = correct_matrix[b]

            sep_pos = (tgt == self.sep_token).nonzero(as_tuple=False).view(-1)
            eos_pos = (tgt == self.eos_token).nonzero(as_tuple=False).view(-1)
            if sep_pos.numel() < 2 or eos_pos.numel() < 1:
                continue

            sep1 = sep_pos[0].item()
            sep2 = sep_pos[1].item()
            eos  = eos_pos[0].item()

            # edge1
            edge_all_total += 1.0
            edge1_ok = cor[sep1 + 1: sep2].all().item() if (sep2 > sep1 + 1) else True
            if edge1_ok:
                edge_all_correct += 1.0

            # edge2
            has_edge2 = (eos > sep2 + 1)
            if has_edge2:
                edge_all_total += 1.0
                edge2_ok = cor[sep2 + 1: eos].all().item()
                if edge2_ok:
                    edge_all_correct += 1.0

        return edge_all_correct, edge_all_total






class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, hid_dim, dropout= 0.1, max_len = 512, batch_first = True):
        """
        Positional Encoding for Transformers.
        
        Args:c
            d_model (int): Embedding dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
            batch_first (bool): Whether input has batch dimension first (batch, seq, embed).
        """
        super().__init__()
        self.batch_first = batch_first
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(p=dropout) if hid_dim > 1 else None  # remove Dropout if d_model=1

        # Embedding layer (only if hid_dim > 1)
        self.embedding = nn.Linear(1, hid_dim) if hid_dim > 1 else None

        # Positional Encoding (only if hid_dim > 1)
        if hid_dim > 1:
            position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.linspace(0, -math.log(10000.0), hid_dim // 2))
            pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
            pe = pe.unsqueeze(1)  # Shape: [max_len, 1, hid_dim]
            self.register_buffer('pe', pe)
        else:
            self.pe = None  # No positional encoding needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hid_dim] (if batch_first)
                              or [seq_len, batch_size, hid_dim] otherwise.
        Returns:
            torch.Tensor: Processed tensor with embedding (if applicable) and positional encoding.
        """
        # Apply embedding layer if it exists
        if self.embedding is not None:
            x = self.embedding(x)


        # Apply positional encoding if applicable
        if self.pe is not None:
            seq_len = x.shape[1] if self.batch_first else x.shape[0]

            assert seq_len <= self.pe.shape[0], f"Sequence length {seq_len} exceeds max_len {self.pe.shape[0]}"

            if self.batch_first:
                x = x + self.pe[:seq_len].transpose(0, 1)  # Shape: [batch, seq, d_model]
            else:
                x = x + self.pe[:seq_len]  # Shape: [seq, batch, d_model]

        # remove Dropout if d_model=1
        return self.dropout(x) if self.dropout is not None else x

        
class Encoder(nn.Module):
    """
    Transformer Encoder with 3 layers.

    Args:
        config (dict): Configuration dictionary containing:
            - hid_dim (int): Hidden dimension size.
            - num_heads (int): Number of attention heads.
            - ff_dim (int): Feedforward layer dimension.
            - dropout (float): Dropout rate.
            - num_layers (int): Number of transformer encoder layers (default: 3).
    """
    def __init__(self, config):
        super().__init__()
        self.hid_dim = config['hid_dim']
        self.num_heads = config.get('num_heads', 8)  # Multi-head attention
        self.ff_dim = config.get('ff_dim', self.hid_dim * 4)  # Feedforward dimension
        self.dropout = config.get('dropout', 0.1)
        self.num_layers = config.get('num_layers', 3)  # Default: 3 layers

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True  # Ensuring batch-first format
        )

        # Transformer Encoder1
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)


        

    def forward(self, x, mask=None):
        """
        Forward pass for Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hid_dim].
            mask (torch.Tensor, optional): Mask tensor for attention (default: None).

        Returns:
            torch.Tensor: Encoded output [batch_size, seq_len, hid_dim].
        """
        return self.encoder(x, mask=mask)




class TokenEmbedding(nn.Module):
    def __init__(self, total_vocab_size, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(total_vocab_size, hid_dim)

    def forward(self, tokens):  # [B, T]
        return self.embedding(tokens)  # [B, T, hid_dim]



class PositionalEncodingOnly(nn.Module):
    def __init__(self, hid_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hid_dim, 2) * (-math.log(10000.0) / hid_dim))
        pe = torch.zeros(max_len, hid_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape [1, max_len, hid_dim]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

        

class TransformerDecoder(nn.Module):
    def __init__(self, hid_dim, num_layers=3, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or hid_dim * 4
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                             )


class Decoder(nn.Module):
    """
    Transformer Decoder module for Seq2Seq generation.
    Implements standard forward, greedy/sampling generation, 
    and beam search with structural masking.

    Args:
        encoder: The Encoder module.
        hid_dim (int): Hidden dimension size.
        total_vocab_size (int): Total size of the output vocabulary.
        sos_token (int): Start-of-Sequence token ID.
        eos_token (int): End-of-Sequence token ID.
        topo_size (int): Vocabulary size for Topology tokens (for masking).
        node_size (int): Vocabulary size for Node tokens (for masking).
        edge_size (int): Vocabulary size for Edge tokens (for masking).
        decoder_layers (int, optional): Number of Decoder layers (default: 3).
        num_heads (int, optional): Number of attention heads (default: 8).
        max_len (int, optional): Max sequence length for positional encoding (default: 512).
    """
    def __init__(self, config, encoder, hid_dim, total_vocab_size, pad_token, sos_token, eos_token, sep_token,
                 decoder_layers=3, num_heads=8, max_len=512,
                
                
                ):
        super().__init__()
        self.encoder = encoder
        self.pos_encoder = EmbeddingWithPositionalEncoding(hid_dim, max_len=max_len) #(B, L ,hid_dim)
        
        self.token_embedding = TokenEmbedding(total_vocab_size, hid_dim)
        self.pos_decoder = PositionalEncodingOnly(hid_dim)
        self.decoder = TransformerDecoder(hid_dim, num_layers=decoder_layers, num_heads=num_heads)

        self.output_head = nn.Linear(hid_dim, total_vocab_size)  # shared output head

        self.encode_dict = MOF_ENCODE_DICT
        self.hid_dim = hid_dim
        self.total_vocab_size = total_vocab_size
        self.sos_token = sos_token
        self.eos_token =eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        
        self.topo_size = config['topo_size']
        self.cn_size  = config['cn_size']
        self.metal_node_size = config['metal_node_size'] 
        self.metal_edge_size = config['metal_edge_size']
        self.selfies_size = config['selfies_size']
        
        special_size = 4

        self.topo_start = special_size
        self.topo_end   = self.topo_start + self.topo_size

        self.node_start = self.topo_end
        self.node_end   = self.node_start + self.metal_node_size          # Nxx “single tkns”

        self.edge_start = self.node_end
        self.edge_end   = self.edge_start + self.metal_edge_size          # Exx “single tkns”

        self.selfies_start = self.edge_end
        self.selfies_end   = self.selfies_start + self.selfies_size # SELFIES vocab size

        self.cn_start = self.selfies_end
        self.cn_end   = self.cn_start + self.cn_size                # CN vocab size (e.g: [M_CN_2]...[O_CN_8] etc)

        self.id2token = {v: k for k, v in self.encode_dict.items()}   # id -> string
        self.topo_ids = torch.arange(self.topo_start, self.topo_end)  # vocab 상 topo id range
        self.selfies_ids = torch.arange(self.selfies_start, self.selfies_end)  # selfies range

        # [Lr] 토큰 id (cn counting용)
        self.lr_id = self.encode_dict["[Lr]"]

        # CN token ids: node/edge CN 전체를 빠르게 판별하려면 (선택)
        self.cn_ids = torch.arange(self.cn_start, self.cn_end)

    def forward(self, desc, target_input):
        """
        Standard forward pass for training.

        Args:
            desc (torch.Tensor): Encoded source sequence tokens [B, D_len].
            target_input (torch.Tensor): Target sequence tokens (shifted right) [B, T_len].

        Returns:
            torch.Tensor: Logits over the vocabulary [B, T_len, V].
        """
        x = self.pos_encoder(desc)
        memory = self.encoder(x)  # [B, D, hid_dim]
        
        tgt_embed = self.token_embedding(target_input)  # [B, T, hid_dim]
        tgt_embed = self.pos_decoder(tgt_embed)

        T = target_input.size(1)
        #tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(desc.device)
        tgt_mask = torch.triu(
                torch.ones(T, T, device=desc.device, dtype=torch.bool),
                diagonal=1
            )
        tgt_key_padding_mask = (target_input == self.pad_token)

        decoder_output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [B, T, hid_dim]
        logits = self.output_head(decoder_output)  # [B, T, total_vocab_size]
        
        return logits

    
    def get_decoder_output(self, desc, target_input):
        """
        Returns the raw hidden states from the decoder.

        Args:
            desc (torch.Tensor): Source tokens [B, D_len].
            target_input (torch.Tensor): Target tokens [B, T_len].
        
        Returns:
            torch.Tensor: Hidden states [B, T_len, hid_dim].
        """
        x = self.pos_encoder(desc)
        memory = self.encoder(x)  # [B, D, hid_dim]
        
        tgt_embed = self.token_embedding(target_input)  # [B, T, hid_dim]
        tgt_embed = self.pos_decoder(tgt_embed)

        T = target_input.size(1)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(desc.device)
        tgt_mask = torch.triu(
                torch.ones(T, T, device=desc.device, dtype=torch.bool),
                diagonal=1
            )
        tgt_key_padding_mask = (target_input == self.pad_token)

        decoder_output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [B, T, hid_dim]

        
        return decoder_output

    



    def generate(self, desc, pad_token, eos_token, max_token_len=512, temperature=1 ):
        """
        Greedy (temp=1.0) or sampling (temp!=1.0) generation.

        Args:
            desc (torch.Tensor): Input description tokens [B, D_len].
            pad_token (int): PAD token ID.
            eos_token (int): EOS token ID.
            max_token_len (int, optional): Max tokens to generate (default: 6).
            temperature (float, optional): Sampling temperature (default: 1).
        
        Returns:
            torch.Tensor: Generated sequence (no SOS) [B, L].
        """
        self.eval()
        B = desc.size(0)
        device = desc.device
    
        x = self.pos_encoder(desc)
        memory = self.encoder(x)
    
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
    
        for _ in range(max_token_len):
            tgt_embed = self.token_embedding(generated)
            tgt_embed = self.pos_decoder(tgt_embed)
    
            T = generated.size(1)
            # tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(desc.device)
            tgt_mask = torch.triu(
                    torch.ones(T, T, device=desc.device, dtype=torch.bool),
                    diagonal=1
                )
            tgt_key_padding_mask = (generated == pad_token)   # [B, T]


            decoder_output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,)
            logits = self.output_head(decoder_output[:, -1])
            
            if temperature == 1.0:
                # Deterministic: pick the highest probability token
                next_token = torch.argmax(logits, dim=-1)
            else:
                # Probabilistic: sample from temperature-adjusted distribution
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    
            next_token[finished] = pad_token  # Overwrite tokens after EOS with PAD
            finished |= (next_token == eos_token)  # Update sample upon finding EOS
    
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    
            if finished.all():
                break


        cur_len = generated.size(1) - 1  
        if cur_len < max_token_len:
            pad_len = max_token_len - cur_len
            pad_block = torch.full((B, pad_len), pad_token, dtype=torch.long, device=device)
            generated = torch.cat([generated, pad_block], dim=1)
    
        return generated[:, 1:]  # remove <sos>

    def generate_with_hidden(self, desc, pad_token, eos_token, max_token_len=512, temperature=1):
        """
        Generation that returns the sequence and the hidden state of each generated token.

        Args:
            desc (torch.Tensor): Input description tokens [B, D_len].
            pad_token (int): PAD token ID.
            eos_token (int): EOS token ID.
            max_token_len (int, optional): Max tokens to generate (default: 6).
            temperature (float, optional): Sampling temperature (default: 1).

        Returns:
            tuple: (sequence [B, L], hidden states [B, L, hid_dim]).
        """
        self.eval()
        B = desc.size(0)
        device = desc.device
    
        x = self.pos_encoder(desc)
        memory = self.encoder(x)
    
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
    
        all_decoder_outputs = []
    
        for _ in range(max_token_len):
            tgt_embed = self.token_embedding(generated)
            tgt_embed = self.pos_decoder(tgt_embed)
    
            T = generated.size(1)
            # tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(desc.device)
            tgt_mask = torch.triu(
                    torch.ones(T, T, device=desc.device, dtype=torch.bool),
                    diagonal=1
                )
            tgt_key_padding_mask = (generated == pad_token)   # [B, T]

            decoder_output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
            # print(decoder_output.shape)
            all_decoder_outputs.append(decoder_output[:, -1:, :])  # Save the hidden state of the final token only
    
            logits = self.output_head(decoder_output[:, -1])
            
            if temperature == 1.0:
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    
            next_token[finished] = pad_token
            finished |= (next_token == eos_token)
    
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    
        decoder_hidden = torch.cat(all_decoder_outputs, dim=1)  # [B, max_len, hidden_dim]
        return generated[:, 1:], decoder_hidden 




    @torch.no_grad()
    def generate_beam(
        self,
        desc: torch.Tensor,
        beam_width: int = 5,
        temperature: float = 1.0,
        max_token_len: int = 512,

        # NEW: GNMT length penalty 옵션
        use_gnmt_length_penalty: bool = False,
        length_penalty_alpha: float = 0.6,
    ):
        self.eval()
        device = desc.device
        B = desc.size(0)
        V = self.total_vocab_size

        # ------------------------------------------------------------
        # 1) Encoder
        # ------------------------------------------------------------
        x = self.pos_encoder(desc)
        memory = self.encoder(x)  # [B, S, H]
        memory_key_padding_mask = None  # desc=[B,183,1] 고정 feature라면 보통 None

        # ------------------------------------------------------------
        # 2) Precompute causal mask (float32 권장)
        # ------------------------------------------------------------
        full_tgt_mask = torch.triu(
            torch.full((max_token_len, max_token_len), float("-inf"), device=device, dtype=torch.float32),
            diagonal=1
        )

        def gnmt_lp(lengths: torch.Tensor) -> torch.Tensor:
            # lengths: (B, beam) or (B, beam, V) broadcastable
            # lp = ((5 + L) / 6) ** alpha
            return ((5.0 + lengths.float()) / 6.0).pow(length_penalty_alpha)

        # ------------------------------------------------------------
        # 3) Helper: next_logits
        # ------------------------------------------------------------
        def next_logits(seq_flat, mem_flat, mem_pad_mask_flat):
            # seq_flat: [B*beam, T]
            Tcur = seq_flat.size(1)

            tgt = self.token_embedding(seq_flat)  # (N, T, H)

            # pos emb: [1,512,256] -> [1,T,H] broadcast
            pos_emb = self.pos_decoder.pe[:, :Tcur, :].to(device=tgt.device, dtype=tgt.dtype)
            tgt = tgt + pos_emb
            tgt = self.pos_decoder.dropout(tgt)

            tgt_mask = full_tgt_mask[:Tcur, :Tcur]  # (T,T) float32
            tgt_key_padding_mask = (seq_flat == self.pad_token)  # (N,T) bool

            decoder_kwargs = {
                "tgt_mask": tgt_mask,
                "tgt_key_padding_mask": tgt_key_padding_mask,
            }
            if mem_pad_mask_flat is not None:
                decoder_kwargs["memory_key_padding_mask"] = mem_pad_mask_flat

            out = self.decoder(tgt, mem_flat, **decoder_kwargs)  # (N,T,H)

            # 마지막 non-pad 위치
            valid_mask = (seq_flat != self.pad_token)
            lengths = valid_mask.long().sum(dim=1)
            last_valid_idx = (lengths - 1).clamp_min(0)

            last_out = out[torch.arange(seq_flat.size(0), device=device), last_valid_idx]
            return self.output_head(last_out)  # (N,V)

        # ------------------------------------------------------------
        # 4) Beam step (length penalty + EOS 확장 차단)
        # ------------------------------------------------------------
        def beam_step(seqs, scores_raw, lengths, mem_use, mem_pad_mask):
            """
            seqs:      (B, beam, T)
            scores_raw:(B, beam)   누적 logp (penalty 미적용 raw)
            lengths:   (B, beam)   생성 길이(= SOS 제외 토큰 수). EOS 후에는 증가 멈춤.
            """
            Bmem, beam, Tcur = seqs.shape
            flatN = Bmem * beam

            seq_flat = seqs.reshape(flatN, Tcur)

            # expand memory
            mem_flat = (
                mem_use.unsqueeze(1)
                .expand(Bmem, beam, mem_use.size(1), mem_use.size(2))
                .reshape(flatN, mem_use.size(1), mem_use.size(2))
            )

            mem_pad_mask_flat = None
            if mem_pad_mask is not None:
                mem_pad_mask_flat = (
                    mem_pad_mask.unsqueeze(1)
                    .expand(Bmem, beam, mem_pad_mask.size(1))
                    .reshape(flatN, mem_pad_mask.size(1))
                )

            logits = next_logits(seq_flat, mem_flat, mem_pad_mask_flat)  # (flatN,V)

            # --- EOS 확장 차단 ---
            # 현재 빔이 EOS로 끝났으면, 다음 토큰은 EOS만 가능하게(다른 토큰 확장 불가)
            last_tok = seq_flat[:, -1]
            is_done = (last_tok == self.eos_token)
            if is_done.any():
                logits[is_done, :] = -float("inf")
                logits[is_done, self.eos_token] = 0.0

            logp = torch.log_softmax(logits / temperature, dim=-1)  # (flatN,V)
            logp = logp.view(Bmem, beam, V)  # (B,beam,V)

            # 후보 raw score
            cand_scores_raw = scores_raw.unsqueeze(-1) + logp  # (B,beam,V)

            # 후보 length 업데이트(선택 기준에 사용)
            # length는 "이번에 한 토큰 생성하면 +1"이 기본인데,
            # 이미 done인 빔(EOS 끝)은 length 증가시키지 않음.
            lengths_exp = lengths.unsqueeze(-1).expand(Bmem, beam, V)  # (B,beam,V)
            inc = torch.ones_like(lengths_exp)

            # done인 빔은 증가 0
            if is_done.any():
                done3 = is_done.view(Bmem, beam, 1).expand(Bmem, beam, V)
                inc = torch.where(done3, torch.zeros_like(inc), inc)

            cand_lengths = lengths_exp + inc  # (B,beam,V)

            # --- Top-k 선택 기준 점수 ---
            if use_gnmt_length_penalty:
                lp = gnmt_lp(cand_lengths)  # (B,beam,V)
                cand_scores_select = cand_scores_raw / lp
            else:
                cand_scores_select = cand_scores_raw

            cand_scores_select_flat = cand_scores_select.view(Bmem, -1)  # (B, beam*V)
            topk_select, topk_idx = torch.topk(cand_scores_select_flat, beam_width, dim=-1)

            beam_idx = topk_idx // V     # (B,beam_width)
            tok_idx  = topk_idx % V      # (B,beam_width)

            # 시퀀스/score/raw/lengths 갱신
            # prefix 선택
            new_seqs = seqs.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, Tcur))  # (B,beam_width,Tcur)
            new_seqs = torch.cat([new_seqs, tok_idx.unsqueeze(-1)], dim=-1)         # (B,beam_width,Tcur+1)

            # raw score는 cand_scores_raw에서 동일 인덱스로 뽑기
            cand_scores_raw_flat = cand_scores_raw.view(Bmem, -1)
            new_scores_raw = cand_scores_raw_flat.gather(1, topk_idx)  # (B,beam_width)

            # lengths도 cand_lengths에서 뽑기
            cand_lengths_flat = cand_lengths.view(Bmem, -1)
            new_lengths = cand_lengths_flat.gather(1, topk_idx)  # (B,beam_width)

            return new_seqs, new_scores_raw, new_lengths

        # ------------------------------------------------------------
        # 5) Init
        # ------------------------------------------------------------
        seqs = torch.full((B, 1, 1), self.sos_token, dtype=torch.long, device=device)
        scores_raw = torch.zeros((B, 1), device=device)
        lengths = torch.zeros((B, 1), dtype=torch.long, device=device)  # SOS 제외 길이

        # 첫 스텝
        seqs, scores_raw, lengths = beam_step(seqs, scores_raw, lengths, memory, memory_key_padding_mask)

        # ------------------------------------------------------------
        # 6) Main loop
        # ------------------------------------------------------------
        for _ in range(max_token_len - 1):
            # 모든 빔이 EOS로 끝났으면 종료
            if (seqs[:, :, -1] == self.eos_token).all():
                break
            seqs, scores_raw, lengths = beam_step(seqs, scores_raw, lengths, memory, memory_key_padding_mask)

        # ------------------------------------------------------------
        # 7) Return: top beam_width 전체
        # ------------------------------------------------------------
        final_seqs = seqs[:, :, 1:]  # SOS 제거

        # pad/truncate to max_token_len
        cur_len = final_seqs.size(-1)
        if cur_len < max_token_len:
            pad_len = max_token_len - cur_len
            pad_tensor = torch.full(final_seqs.shape[:-1] + (pad_len,), self.pad_token, dtype=torch.long, device=device)
            final_seqs = torch.cat([final_seqs, pad_tensor], dim=-1)
        else:
            final_seqs = final_seqs[..., :max_token_len]

        # 선택된 top5를 “정렬 점수”로도 같이 주고 싶으면:
        if use_gnmt_length_penalty:
            # 최종 출력 기준 점수(정렬 기준) = raw / lp(length)
            final_scores = scores_raw / gnmt_lp(lengths)
        else:
            final_scores = scores_raw

        # final_seqs: (B, beam_width, max_token_len)
        # scores_raw: (B, beam_width)  누적 logp
        # final_scores: (B, beam_width) penalty 적용 점수(선택 기준과 일치)
        return final_seqs, scores_raw, final_scores, lengths
    

    @torch.no_grad()
    def generate_beam_session_greedy(
        self,
        desc: torch.Tensor,
        beam_width: int = 5,
        temperature: float = 1.0,
        max_token_len: int = 512,
        max_blocks: int = 4,
   
    ):
        self.eval()
        device = desc.device
        B = desc.size(0)
        V = self.total_vocab_size

        x = self.pos_encoder(desc)
        memory = self.encoder(x)  # [B, S, H]

        # ============================================================
        # <<< NEW >>> util: last valid token/index based on PAD
        # ============================================================
        def last_valid_token(seq: torch.Tensor, pad_token: int):
            """
            seq: [N, T]
            return last_tok: [N], last_idx: [N]
            """
            # PAD가 아닌 위치는 1, PAD는 0
            valid_mask = (seq != pad_token).long()
            
            # [0, 1, 2, ...] 인덱스 배열과 곱해서, PAD가 아닌 가장 큰 인덱스를 찾음
            # 예: [A, B, PAD, C] -> mask [1, 1, 0, 1] -> idx [0, 1, 0, 3] -> argmax=3 (C)
            last_idx = (torch.arange(seq.size(1), device=seq.device) * valid_mask).argmax(dim=1)
            
            # 해당 인덱스의 토큰을 추출
            last_tok = seq.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            return last_tok, last_idx

        # ============================================================
        # <<< CHANGED >>> next_logits: out[:, -1] 금지!
        #              각 row의 last_valid_idx에서 hidden을 뽑아서 logits 계산
        # ============================================================
        def next_logits(seq_flat: torch.Tensor, mem_flat: torch.Tensor, pos_flat: torch.Tensor):
            tgt = self.token_embedding(seq_flat) 
            pe_table = self.pos_decoder.pe.squeeze(0) 
            pos_emb = pe_table[pos_flat] 
            tgt = tgt + pos_emb
            tgt = self.pos_decoder.dropout(tgt) 
            
            Tcur = seq_flat.size(1)
            tgt_mask = torch.triu(torch.ones(Tcur, Tcur, device=device, dtype=torch.bool), diagonal=1)
            tgt_key_padding_mask = (seq_flat == self.pad_token)

            out = self.decoder(
                tgt, mem_flat,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # [✅ 완벽한 수정] 개수를 세지 않고, 가장 오른쪽에 있는 진짜 토큰의 인덱스를 직접 찾음
            # 패딩이 아닌 위치는 1, 패딩은 0. 거기에 인덱스 번호를 곱해서 가장 큰 값을 찾으면 그게 마지막 진짜 토큰입니다.
            valid_mask = (seq_flat != self.pad_token).long()
            last_valid_idx = (torch.arange(Tcur, device=device) * valid_mask).argmax(dim=1)
            
            # 각 빔에서 진짜 마지막 위치의 출력만 뽑아냄
            last_out = out[torch.arange(seq_flat.size(0), device=device), last_valid_idx]
            
            return self.output_head(last_out)

        # ============================================================
        # <<< CHANGED >>> beam_expand: done 판정도 last_valid_token 기반
        # ============================================================
        def beam_expand(seqs, scores, pos_ids, k, mem_use):
            Bmem, beam, Tcur = seqs.shape
            if Tcur >= max_token_len - 1:
                return seqs, scores, pos_ids

            flatN = Bmem * beam
            seq_flat = seqs.reshape(flatN, Tcur)
            pos_flat = pos_ids.reshape(flatN, Tcur)

            mem_flat = mem_use.unsqueeze(1).expand(Bmem, beam, mem_use.size(1), mem_use.size(2)) \
                            .reshape(flatN, mem_use.size(1), mem_use.size(2))

            logits = next_logits(seq_flat, mem_flat, pos_flat)
            
            # [✅ 완벽한 수정] 여기도 동일하게 가장 오른쪽 진짜 토큰을 찾아서 검사
            valid_mask = (seq_flat != self.pad_token).long()
            last_valid_idx = (torch.arange(Tcur, device=device) * valid_mask).argmax(dim=1)
            real_last_tokens = seq_flat[torch.arange(flatN, device=device), last_valid_idx]

            # 진짜 마지막이 EOS인 경우에만 종료. (PAD에 속지 않음)
            is_done = (real_last_tokens == self.eos_token)
            if is_done.any():
                logits[is_done, :] = -float('inf')
                logits[is_done, self.pad_token] = 0.0
                
            logp = torch.log_softmax(logits / temperature, dim=-1)
            logp = logp.view(Bmem, beam, V)

            total = scores.unsqueeze(-1) + logp
            total = total.view(Bmem, -1)

            topk_scores, topk_idx = torch.topk(total, k, dim=-1)
            beam_idx = topk_idx // V
            tok_idx = topk_idx % V

            new_seqs = seqs.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, Tcur))
            new_seqs = torch.cat([new_seqs, tok_idx.unsqueeze(-1)], dim=-1)

            new_pos = pos_ids.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, Tcur))
            
            # [✅ 중요] 새 토큰의 위치 ID는 직전 '진짜 토큰'의 위치 + 1 이어야 함
            last_valid_pos = pos_flat[torch.arange(flatN, device=device), last_valid_idx].view(Bmem, beam)
            last_valid_pos_gathered = last_valid_pos.gather(1, beam_idx).unsqueeze(-1)
            next_pos_val = last_valid_pos_gathered + 1
            
            new_pos = torch.cat([new_pos, next_pos_val], dim=-1)

            return new_seqs, topk_scores, new_pos

        # ============================================================
        # <<< CHANGED >>> greedy_until_sep: done/stop도 last_valid_token 기반 + logits 위치도 안전
        # ============================================================
        def greedy_until_sep(seqs, scores, pos_ids, mem_sel, max_steps):
            for _ in range(max_steps):
                B2, beam2, Tcur = seqs.shape
                if Tcur >= max_token_len - 1:
                    break

                flatN = B2 * beam2
                seq_flat = seqs.reshape(flatN, Tcur)
                pos_flat = pos_ids.reshape(flatN, Tcur)

                mem_flat = mem_sel.unsqueeze(1).expand(B2, beam2, mem_sel.size(1), mem_sel.size(2)) \
                                .reshape(flatN, mem_sel.size(1), mem_sel.size(2))

                logits = next_logits(seq_flat, mem_flat, pos_flat)

                last_tok, _ = last_valid_token(seq_flat, self.pad_token)
                is_done = (last_tok == self.sep_token) | (last_tok == self.eos_token) | (last_tok == self.pad_token)
                if is_done.any():
                    logits[is_done, :] = -float("inf")
                    logits[is_done, self.pad_token] = 0.0

                logp = torch.log_softmax(logits / temperature, dim=-1)
                next_tok = torch.argmax(logp, dim=-1)
                next_logp = logp.gather(1, next_tok.unsqueeze(1)).squeeze(1)

                next_tok = next_tok.view(B2, beam2)
                next_logp = next_logp.view(B2, beam2)

                seqs = torch.cat([seqs, next_tok.unsqueeze(-1)], dim=-1)
                scores = scores + next_logp

                next_pos_val = pos_ids[:, :, -1:] + 1
                pos_ids = torch.cat([pos_ids, next_pos_val], dim=-1)

                # <<< CHANGED >>> 종료도 next_tok 기준(방금 붙인 토큰이 SEP/EOS/PAD면 종료)
                is_finished = (next_tok == self.sep_token) | (next_tok == self.eos_token) | (next_tok == self.pad_token)
                if is_finished.all():
                    break

            return seqs, scores, pos_ids

        # ============================================================
        # initialization
        # ============================================================
        seqs = torch.full((B, 1, 1), self.sos_token, dtype=torch.long, device=device)
        scores = torch.zeros((B, 1), device=device)
        pos_ids = torch.zeros((B, 1, 1), dtype=torch.long, device=device)

        # Topology
        seqs, scores, pos_ids = beam_expand(seqs, scores, pos_ids, k=beam_width, mem_use=memory)

        # ============================================================
        # blocks
        # ============================================================
        for _ in range(max_blocks):
            # <<< CHANGED >>> "모두 종료" 판단도 last_valid_token 기반으로 해야 함
            flat = seqs.view(B * beam_width, -1)
            last_tok, _ = last_valid_token(flat, self.pad_token)
            if ((last_tok == self.eos_token) | (last_tok == self.pad_token)).all():
                break

            if seqs.size(-1) >= max_token_len - 1:
                break

            # CN
            seqs, scores, pos_ids = beam_expand(seqs, scores, pos_ids, k=beam_width, mem_use=memory)
            if seqs.size(-1) >= max_token_len - 1:
                break

            # first content
            seqs, scores, pos_ids = beam_expand(seqs, scores, pos_ids, k=beam_width, mem_use=memory)
            if seqs.size(-1) >= max_token_len - 1:
                break

            # SELFIES stream (greedy until SEP/EOS/PAD)
            # <<< CHANGED >>> need도 last_valid_token 기반
            flat = seqs.view(B * beam_width, -1)
            last_tok, _ = last_valid_token(flat, self.pad_token)
            need = (last_tok != self.sep_token) & (last_tok != self.eos_token) & (last_tok != self.pad_token)

            if need.any():
                idx = need.nonzero(as_tuple=False).squeeze(1)

                flat_pos = pos_ids.view(B * beam_width, -1)
                flat_scores = scores.view(B * beam_width)

                sub_seqs = flat[idx].unsqueeze(1)        # [n,1,T]
                sub_pos = flat_pos[idx].unsqueeze(1)
                sub_scores = flat_scores[idx].unsqueeze(1)

                b_idx = (idx // beam_width).long()
                mem_sel = memory[b_idx]

                sub_seqs, sub_scores, sub_pos = greedy_until_sep(
                    sub_seqs, sub_scores, sub_pos, mem_sel, max_steps=max_selfies_len
                )

                new_sub = sub_seqs[:, 0, :]
                new_sub_pos = sub_pos[:, 0, :]
                new_sub_scores = sub_scores[:, 0]

                # --- 길이 정렬은 "다음 연산을 위해" 최소한으로만 필요 ---
                # <<< CHANGED >>> 여기서도 PAD는 어쩔 수 없이 맞추되,
                #                마지막 유효 토큰 로직이 있으니 seqs[:,:,-1]에 의존하지 않음.
                maxL = max(flat.size(1), new_sub.size(1))
                if maxL > max_token_len - 1:
                    maxL = max_token_len - 1

                # pad flat
                if flat.size(1) < maxL:
                    pad_len = maxL - flat.size(1)
                    pad_seq = torch.full((flat.size(0), pad_len), self.pad_token, dtype=torch.long, device=device)
                    flat = torch.cat([flat, pad_seq], dim=1)

                    last_pos = flat_pos[:, -1:].expand(-1, pad_len)
                    flat_pos = torch.cat([flat_pos, last_pos], dim=1)

                flat = flat[:, :maxL]
                flat_pos = flat_pos[:, :maxL]

                # pad new_sub
                if new_sub.size(1) < maxL:
                    pad_len = maxL - new_sub.size(1)
                    pad_seq = torch.full((new_sub.size(0), pad_len), self.pad_token, dtype=torch.long, device=device)
                    new_sub = torch.cat([new_sub, pad_seq], dim=1)

                    last_pos = new_sub_pos[:, -1:].expand(-1, pad_len)
                    new_sub_pos = torch.cat([new_sub_pos, last_pos], dim=1)

                new_sub = new_sub[:, :maxL]
                new_sub_pos = new_sub_pos[:, :maxL]

                flat[idx] = new_sub
                flat_pos[idx] = new_sub_pos
                flat_scores[idx] = new_sub_scores

                seqs = flat.view(B, beam_width, -1)
                pos_ids = flat_pos.view(B, beam_width, -1)
                scores = flat_scores.view(B, beam_width)

        # ============================================================
        # <<< CHANGED >>> 세션 종료: 마지막 유효 토큰 기준으로 EOS를 "예측"하게 한 스텝 더 진행
        #                (PAD 때문에 seqs[:,:,-1] 쓰면 안 됨)
        # ============================================================
        if seqs.size(-1) < max_token_len - 1:
            flat = seqs.view(B * beam_width, -1)
            last_tok, _ = last_valid_token(flat, self.pad_token)

            need_eos = (last_tok != self.eos_token) & (last_tok != self.pad_token)
            # (여기서 last_tok == SEP 인 애들도 EOS로 끝내고 싶으니 포함되는 게 맞음)
            if need_eos.any():
                seqs, scores, pos_ids = beam_expand(seqs, scores, pos_ids, k=beam_width, mem_use=memory)

        # return (drop SOS)
        # 1. SOS 토큰 제거 (생성 결과만 남김)
        if beam_width == 1:
            final_seqs = seqs[:, 0, 1:]
        else:
            final_seqs = seqs[:, :, 1:]
            
        # 2. [핵심] 현재 길이가 512(max_token_len)보다 작으면 무조건 PAD로 채움
        current_len = final_seqs.size(-1)
        if current_len < max_token_len:
            pad_len = max_token_len - current_len
            # final_seqs의 앞쪽 차원(Batch, Beam)은 그대로 두고, 마지막 차원만 pad_len으로 생성
            pad_tensor = torch.full(
                final_seqs.shape[:-1] + (pad_len,), 
                self.pad_token, 
                dtype=torch.long, 
                device=device
            )
            final_seqs = torch.cat([final_seqs, pad_tensor], dim=-1)
        
        # (혹시 모를 오버플로우 대비 안전장치)
        elif current_len > max_token_len:
            final_seqs = final_seqs[..., :max_token_len]

        return final_seqs

 

def topk_token_metrics(logits, target_output, pad_token, k=5):
    """
    logits: [B, T, V]
    target_output: [B, T]
    """
    with torch.no_grad():
        mask = (target_output != pad_token)            # only valid token
        topk_idx = logits.topk(k, dim=-1).indices      # [B, T, k]
        # Check if the ground truth is within top-k at each position
        correct_topk = (topk_idx == target_output.unsqueeze(-1)).any(dim=-1)  # [B, T]
        correct_topk = correct_topk & mask

        # token-level top-k accuracy
        tok_acc_topk = correct_topk.sum().float() / mask.sum().float().clamp_min(1)

        # Sequence-level (whether all valid positions achieved top-k accuracy)
        seq_all_topk = (correct_topk | ~mask).all(dim=1).float().mean()

    return tok_acc_topk, seq_all_topk, correct_topk, mask



