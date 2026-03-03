import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer
from utils import create_graph, Collator
from torch.utils.data import DataLoader
import spacy

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import hydra
from utils import assign_distance_bucket  # 你的工具函数
import logging
log = logging.getLogger(__name__)

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

class DocRED(Dataset):
    def __init__(self, data_module, dataset_dir: str, file_name: str, tokenizer: PreTrainedTokenizer):
        super(DocRED, self).__init__()
        self.data_module = data_module
        self.name = "re-docred"
        dataset_dir = Path(dataset_dir)
        save_dir = dataset_dir / "bin"
        with open(dataset_dir / "rel2id.json", "r", encoding="utf-8") as f:
            self.rel2id: Dict[str, int] = json.load(f)
        with open(dataset_dir / "ner2id.json", "r", encoding="utf-8") as f:
            self.ner2id: Dict[str, int] = json.load(f)
        self.id2rel = {value: key for key, value in self.rel2id.items()}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_name_or_path = tokenizer.name_or_path
        model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
        ori_path = str(dataset_dir / file_name)
        with open(ori_path, "r",encoding='utf-8') as fh:
            self.data: List[Dict] = json.load(fh)
        split = ori_path[ori_path.rfind("/") + 1:ori_path.rfind(".")]
        save_path = save_dir / (split + f".{model_name_or_path}.pt")
        self.rel_code2name = self._get_rel_code_to_name()
        if os.path.exists(save_path):
            print(f"Loading CDR {split} features ...")
            self.features = torch.load(save_path)
        else:
            self.features = self.read_docred(split, tokenizer)
            torch.save(self.features, save_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def read_docred(self, split, tokenizer):
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []

        entity_1 = []
        entity_2 = []
        entity_3 = []
        entity_4 = []

        max_tokens_len = 0

        for docid, doc in tqdm(enumerate(self.data), desc=f"Reading CDR {split} data", total=len(self.data), ncols=100):
            delete_doc = [17222831, 11318962, 12073281, 17559688, 18312663, 20619739, 12398019, 18385788, 11911406]
            if doc['title'] in delete_doc:
                continue
            doc_title = doc['title']
            title: str = doc['title']
            entities: List[List[Dict]] = doc['vertexSet']
            sentences: List[List[str]] = doc['sents']
            ori_labels: List[Dict] = doc.get('labels', [])

            ENT_NUM = len(entities)
            SENT_NUM = len(sentences)
            MEN_NUM = len([m for e in entities for m in e if "coref" not in m])
            COREF_NUM = len([m for e in entities for m in e if "coref" in m])

            text_parts = []
            for sent in sentences:
                if isinstance(sent, list) and len(sent) > 0:
                    sent_str = " ".join(sent).strip()  # 单词→句子字符串
                    text_parts.append(sent_str)
            text = ". ".join(text_parts).strip()
            text = text + "." if text else f"Empty text for doc_{doc_title}"

            spo_list = []
            ent_index_to_name = {}
            for ent_idx, mentions in enumerate(entities):
                if len(mentions) == 0:
                    ent_name = f"Entity_{ent_idx}"
                else:
                    first_mention = next((m for m in mentions if "coref" not in m), mentions[0])
                    name_val = first_mention.get("name", f"Entity_{ent_idx}")
                    if isinstance(name_val, list):
                        ent_name = " ".join(name_val).strip()
                    else:
                        ent_name = str(name_val).strip()
                    ent_name = ent_name if ent_name else f"Entity_{ent_idx}"
                ent_index_to_name[ent_idx] = ent_name

            for label in ori_labels:
                h_idx = label.get("h", -1)
                t_idx = label.get("t", -1)
                rel_val = label.get("r", "")
                rel_code = str(rel_val).strip()
                dist = label.get("dist", "")

                if h_idx not in ent_index_to_name or t_idx not in ent_index_to_name:
                    continue
                rel_name = self.rel_code2name.get(rel_code, rel_code)

                spo = {
                    "subject": ent_index_to_name[h_idx],
                    "predicate": rel_name,
                    "object": ent_index_to_name[t_idx]
                }
                spo_list.append(spo)

            mention_index = []
            for entity in entities:
                for mention in entity:
                    if "coref" not in mention:
                       mention_index.append((mention["pos"][0], mention["pos"][1]))
            tokens: List[str] = []
            sent_pos = {}
            sent_map = {}
            i_s = 0
            i_t = 0
            for sent in sentences:
                sent_pos[i_s] = len(tokens)
                for word in sent:
                    word_tokens = tokenizer.tokenize(word)
                    for start, end in mention_index:
                        if start == i_t:
                            word_tokens = ["*"] + word_tokens
                        if end == i_t + 1:
                            word_tokens = word_tokens + ["*"]
                    sent_map[i_t] = len(tokens)
                    tokens.extend(word_tokens)
                    i_t += 1
                sent_map[i_t] = len(tokens)
                i_s += 1
            sent_pos[i_s] = len(tokens)

            final_sent_pos = []
            for i in range(len(sent_pos)-1):
                final_sent_pos.append((sent_pos[i], sent_pos[i+1]))

            train_triple = {}
            for label in ori_labels:
                h, t, r = label['h'], label['t'], label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

            entity_pos = [[] for _ in range(ENT_NUM)]
            mention_pos: Tuple[List[int], List[int]] = ([], [])

            ent2mention: List[List[int]] = [[] for _ in range(ENT_NUM)]
            mention2ent: List[int] = []

            sent2mention: List[List[int]] = [[] for _ in range(SENT_NUM)]
            mention2sent: List[int] = []

            mention_id = 0
            entity_types = [0, 1]
            for entity_id, entity in enumerate(entities):
                for mention in entity:
                    sent_id, pos = mention["sent_id"], mention["pos"]
                    start, end = pos[0], pos[1]
                    new_start = sent_map[start]
                    new_end = sent_map[end]

                    entity_pos[entity_id].append((new_start, new_end))
                    mention_pos[0].append(new_start)
                    mention_pos[1].append(new_end)

                    ent2mention[entity_id].append(mention_id)
                    mention2ent.append(entity_id)

                    sent2mention[sent_id].append(mention_id)
                    mention2sent.append(sent_id)

                    mention_id += 1

            hts: List[List[int]] = []
            relations: List[List[int]] = []
            dists, ent_dis = [], []
            for (h, t) in train_triple.keys():
                head_entity_pos, tail_entity_pos = entities[h][0]['pos'], entities[t][0]['pos']
                if head_entity_pos[1] < tail_entity_pos[0]:
                    abs_dis = tail_entity_pos[0] - head_entity_pos[1]
                elif head_entity_pos[0] > tail_entity_pos[1]:
                    abs_dis = head_entity_pos[0] - tail_entity_pos[1]
                else:
                    abs_dis = 0
                relation = [0] * len(gda_rel2id)
                for label in train_triple[h, t]:
                    r = label["relation"]
                    relation[r] = 1
                    if label["dist"] == "CROSS":
                        dist = 1
                    elif label["dist"] == "NON-CROSS":
                        dist = 0
                relations.append(relation)
                hts.append([h, t])
                dists.append(dist)
                ent_dis.append(abs_dis)
                pos_samples += 1

            max_tokens_len = max(max_tokens_len, len(tokens) + 2)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            assert len(input_ids) == len(tokens) + 2

            entity_start, entity_end = {}, {}
            for start, end in mention_index:
                entity_start[(start)] = "*"
                entity_end[(end)] = "*"

            words = []
            lengthofPice = 0
            token_map = []
            for i_s, sent in enumerate(sentences):
                for i_t, token in enumerate(sent):
                    oneToken = []
                    words.append(token)
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if any(i_t == item[0] for item in mention_index):
                        tokens_wordpiece = [entity_start[(i_t)]] + tokens_wordpiece
                        oneToken.append(lengthofPice + 1)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice)

                    elif any(i_t == item[1] for item in mention_index):
                        tokens_wordpiece = tokens_wordpiece + [entity_end[(i_t)]]
                        oneToken.append(lengthofPice)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice - 1)
                    else:
                        oneToken.append(lengthofPice)
                        lengthofPice += len(tokens_wordpiece)
                        oneToken.append(lengthofPice)
                    token_map.append(oneToken)

            men_graph = create_graph(mention2ent, ent2mention, sent2mention, mention2sent,
                                                                self.rel2id, MEN_NUM, ENT_NUM, SENT_NUM,
                                                                1)

            i_line += 1
            feature = {
                'title': title,
                'input_ids': input_ids,
                'hts': hts,
                'sent_pos': final_sent_pos,
                'entity_pos': entity_pos,
                'mention_pos': mention_pos[0],
                'entity_types': entity_types,
                'men_graph': men_graph,
                'label': relations,
                'dists': dists,
                'ent_dis': ent_dis,
                'text': text,  # 新增：完整文本
                'spo_list': spo_list,  # 新增：解析后的三元组列表
            }
            features.append(feature)

            #if 1<=len(entity_pos)<5:
            #    entity_1.append(doc)
            #elif 5<=len(entity_pos)<10:
            #    entity_2.append(doc)
            #elif 10<= len(entity_pos)<15:
            #    entity_3.append(doc)
            #else:
            #    entity_4.append(doc)

        #print(len(entity_1),len(entity_2),len(entity_3),len(entity_4))
        #with open("F:\GDA_json格式/first.json", "w") as fh:
        #    json.dump(entity_1, fh)
        #with open("F:\GDA_json格式/second.json", "w") as fh:
        #    json.dump(entity_2, fh)
        #with open("F:\GDA_json格式/third.json", "w") as fh:
        #    json.dump(entity_3, fh)
        #with open("F:\GDA_json格式/fourth.json", "w") as fh:
        #    json.dump(entity_4, fh)

        print("# of documents {}.".format(i_line))
        print("maximum tokens length:", max_tokens_len)
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        return features

    def _get_rel_code_to_name(self) -> Dict[str, str]:

        dataset_type = "CDR"
        dataset_rel_mapping = {
            "CDR": {
                "1": "chemical induced disease",
                "0": "no chemical disease induction relation"
            },
            "GDA": {
                "1": "gene disease association",
                "0": "no gene disease association"
            },
            "CHR": {
                "1": "clinical human relation",
                "0": "no relation"
            }
        }

        if dataset_type in dataset_rel_mapping:
            log.info(f"当前数据集类型：{dataset_type}，使用对应的关系编码→文字映射")
            return dataset_rel_mapping[dataset_type]
        else:
            log.warning(
                f"未知数据集类型：{dataset_type}（仅支持 CDR/GDA/CHR），"
                f"默认使用 CDR 数据集的关系映射"
            )
            return dataset_rel_mapping["CDR"]

class DocREDataModule:
    def __init__(
            self,
            dataset_dir: str,
            tokenizer: PreTrainedTokenizer,
            train_file: str,
            dev_file: str,
            test_file: str,
            train_batch_size: int,
            test_batch_size: int
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_file = train_file

        self.collate_fnt = Collator(tokenizer)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.coref_nlp = spacy.load("en_coreference_web_trf")

        self.data_train = DocRED(self, dataset_dir, train_file, tokenizer)

        self.data_dev = DocRED(self, dataset_dir, dev_file, tokenizer)

        self.data_test = DocRED(self, dataset_dir, test_file, tokenizer)

    @property
    def train_dataset(self):
        return self.data_train

    @property
    def dev_dataset(self):
        return self.data_dev

    @property
    def test_dataset(self):
        return self.data_test
    # dataloader对数据进行预处理
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fnt,
        )

    def dev_dataloader(self):
        return DataLoader(
            dataset=self.data_dev,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./PLM/deberta-v3-large')
    dm = DocREDataModule('./data/CDR', tokenizer, 'train.json', 'dev.json', 'test.json', test_batch_size=2, train_batch_size= False)
    #   dm = DocREDataModule('./data/DocRED', tokenizer, 'train_revised.json', 'dev_revised.json', 'test_revised.json', False, False, False, 2)
    # dm.gen_train_facts()
    # dm.data_train.official_evaluate_benchmark(torch.tensor([]))
