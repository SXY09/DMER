import json
import os

import torch
import random
import numpy as np
import dgl
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import rich
import rich.syntax
import rich.tree
from collections import defaultdict, Counter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    lm_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    return lm_lr, classifier_lr


def print_config_tree(cfg: DictConfig, file=None):
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for filed in cfg:
        branch = tree.add(filed, style=style, guide_style=style)
        config_group = cfg[filed]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree, file=file)

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        titles = [f['title'] for f in batch]
        input_ids = {'input_ids': [f['input_ids'] for f in batch]}
        hts = [f["hts"] for f in batch]
        sent_pos = [f['sent_pos'] for f in batch]
        entity_pos = [f["entity_pos"] for f in batch]
        mention_pos = [torch.tensor(f["mention_pos"]) for f in batch]
        entity_types = [f['entity_types'] for f in batch]
        men_graphs = dgl.batch([f['men_graph'] for f in batch])
        labels = [f["label"] for f in batch]
        dists = [f["dists"] for f in batch]
        ent_dis = [f["ent_dis"] for f in batch]
        texts = [f['text'] for f in batch]
        spo_lists = [f['spo_list'] for f in batch]
        inputs = self.tokenizer.pad(input_ids, return_tensors='pt')
        output = {
            "titles": titles,
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "hts": hts,
            "sent_pos": sent_pos,
            "entity_pos": entity_pos,
            "mention_pos": mention_pos,
            "entity_types": entity_types,
            "men_graphs": men_graphs,
            "labels": labels,
            "dists": dists,
            "ent_dis": ent_dis,
            "texts": texts,
            "spo_lists": spo_lists,
        }
        return output


def create_graph(men2ent, ent2men, sent2men, men2sent, rel2id, MEN_NUM, ENT_NUM, SENT_NUM, DOC_NUM):
    men_graph_dict = {}
    doc_id = list(range(DOC_NUM))  # doc node
    men_ids = list(range(DOC_NUM, DOC_NUM + MEN_NUM))  # mention nodes
    sent_ids = list(range(DOC_NUM + MEN_NUM, DOC_NUM + MEN_NUM + SENT_NUM))  # sent nodes
    men_graph_dict["node", "d-s", "node"] = (doc_id * SENT_NUM + sent_ids, sent_ids + doc_id * SENT_NUM)
    sss = []
    for i in range(SENT_NUM):
        for j in range(i + 1, SENT_NUM):
            sss.append((sent_ids[i], sent_ids[j]))
            sss.append((sent_ids[j], sent_ids[i]))
    men_graph_dict["node", "s-s", "node"] = sss
    men2sent = np.array(sent_ids)[men2sent].tolist()
    men_graph_dict["node", "s-m", "node"] = (men_ids + men2sent, men2sent + men_ids)
    ie_mms = []
    for ems in ent2men:
        n = len(ems)
        for i in range(n):
            for j in range(i + 1, n):
                x, y = ems[i], ems[j]
                ie_mms.append((men_ids[x], men_ids[y]))
                ie_mms.append((men_ids[y], men_ids[x]))
    men_graph_dict["node", "ie/m-m", "node"] = ie_mms
    is_mms = []
    for sms in sent2men:
        n = len(sms)
        for i in range(n):
            for j in range(i + 1, n):
                x, y = sms[i], sms[j]
                is_mms.append((men_ids[x], men_ids[y]))
                is_mms.append((men_ids[y], men_ids[x]))
    men_graph_dict["node", "is/m-m", "node"] = is_mms
    men_graph = dgl.heterograph(men_graph_dict)
    assert men_graph.num_nodes() == DOC_NUM + MEN_NUM + SENT_NUM
    assert men_graph.num_edges("d-s") == SENT_NUM * 2
    assert men_graph.num_edges("s-s") == SENT_NUM * (SENT_NUM - 1)
    assert men_graph.num_edges("s-m") == MEN_NUM * 2

    def fc_edge_nums(gms):
        edge_nums = 0
        for ms in gms:
            gn = len(ms)
            edge_nums += gn * (gn - 1)
        return edge_nums

    assert men_graph.num_edges("is/m-m") == fc_edge_nums(sent2men)
    assert men_graph.num_edges("ie/m-m") == fc_edge_nums(ent2men)
    return men_graph

def assign_distance_bucket(distance, buckets):
    """根据距离分配区间标签"""
    for i, bound in enumerate(buckets):
        if distance < bound:
            return i
    return len(buckets)