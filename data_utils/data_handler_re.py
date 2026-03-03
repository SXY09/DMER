import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional
import logging
import pandas as pd
from datasets import Dataset
import os
from torch.utils.data import Dataset as TorchDataset
from importlib import import_module
import json
from modules.memory.memory import BaseMemory

log = logging.getLogger(__name__)


class DataHandlerRE:
    ds_test: Dataset
    ds_pred: Optional[Dataset]
    ds_index: Dataset
    schema_dict: Dict[str, Dict]
    correct_memory: BaseMemory
    reflexion_memory: BaseMemory
    def __init__(
        self,
        cfg,
        tokenizer: PreTrainedTokenizer,
        datamodule
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.datamodule = datamodule  # 复用外部已处理好的 datamodule
        self.schema_dict = {}
        self.fn_schema = getattr(self.cfg.react_memory.data, "fn_schema", "")
        self.load_schema()
        self.rel2id = self._get_rel_mapping()
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.rel_code2name = self._get_rel_code_to_name()
        self.train_samples = self._extract_core_fields(
            dataset=self.datamodule.train_dataset,
            split="train"
        )
        self.dev_samples = self._extract_core_fields(
            dataset=self.datamodule.dev_dataset,
            split="dev"
        )
        self.test_samples = self._extract_core_fields(
            dataset=self.datamodule.test_dataset,
            split="test"
        )
        self._init_dataset_attributes()
        self._try_load_initial_pred()
        log.info(
            f"Data Loaded: Train Samples={len(self.train_samples)}, "
            f"Dev Samples={len(self.dev_samples)}, "
            f"Test Samples={len(self.test_samples)}"
        )

    def _init_dataset_attributes(self) -> None:
        self.ds_test = Dataset.from_list(self.test_samples)
        log.info(f"ds_test 初始化完成，样本数：{len(self.ds_test)}")
        self.ds_index = Dataset.from_list(self.train_samples)
        log.info(f"ds_index 初始化完成，样本数：{len(self.ds_index)}")
        self.ds_pred = None
        log.info("ds_pred 已初始化（当前为空，需通过预测或加载文件填充）")

    def _try_load_initial_pred(self) -> None:
        try:
            pred_save_path = self.cfg.data.ofn_pred
            if pred_save_path and os.path.exists(pred_save_path):
                self.load_pred_results(pred_save_path)
                log.info(f"从配置路径自动加载预测结果：{pred_save_path}，样本数：{len(self.ds_pred)}")
        except Exception as e:
            log.warning(f"自动加载预测结果失败（可能路径未配置或文件不存在）：{str(e)}")
    def load_pred_results(self, load_path: Optional[str] = None) -> None:
        datasets = import_module("datasets")  # 跳过本地同名文件
        Dataset = datasets.Dataset
        final_path = load_path if load_path else self.cfg.data.ofn_pred
        if not final_path or not os.path.exists(final_path):
            raise FileNotFoundError(f"预测结果文件不存在：{final_path}")
        pred_df = pd.read_json(final_path, lines=True)
        self.ds_pred = Dataset.from_pandas(pred_df)
        log.info(f"已从 {final_path} 加载预测结果，ds_pred 样本数：{len(self.ds_pred)}")

    def _get_rel_mapping(self) -> Dict[str, int]:
        try:
            if hasattr(self.datamodule.train_dataset, "rel2id"):
                return self.datamodule.train_dataset.rel2id
            elif self.cfg.react_memory.data.name == "CDR":
                return {"1:NR:2": 0, "1:CID:2": 1}
            else:
                return {"1:NR:2": 0, "1:GDA:2": 1}
        except Exception as e:
            log.warning(f"获取关系映射时出错: {str(e)}，使用 CDR 数据集默认映射")
            return {"1:NR:2": 0, "1:CID:2": 1}

    def _get_rel_code_to_name(self) -> Dict[str, str]:
        dataset_type = getattr(self.cfg.react_memory.data, "name", "CDR").upper()  # 转大写避免大小写问题（如 cdr→CDR）
        dataset_rel_mapping = {
            "CDR": {
                "1:CID:2": "chemical induced disease",
                "1:NR:2": "no chemical disease induction relation"
            },
            "GDA": {
                "1:GDA:2": "gene disease association",
                "1:NR:2": "no gene disease association"
            },
            "CHR": {
                "1:React:2": "chemical metabolite interaction",
                "1:NR:2": "no chemical metabolite interaction"
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

    def _extract_core_fields(self, dataset, split: str) -> List[Dict]:
        core_samples = []
        # 检查数据集是否包含原始文档（data 属性）
        if not hasattr(dataset, "data"):
            log.error(f"{split} 数据集缺少原始文档（data 属性），无法提取字段")
            return core_samples

        for doc_idx, doc in enumerate(dataset.data):
            try:
                doc_title = doc.get("title", f"{split}_doc_{doc_idx}")
                sents = doc.get("sents", [])
                text_parts = []
                for sent in sents:
                    if isinstance(sent, list) and len(sent) > 0:
                        sent_str = " ".join(sent).strip()
                        text_parts.append(sent_str)
                text = ". ".join(text_parts)
                text = text.strip() + "." if text else f"Empty text for {doc_title}"
                vertex_set = doc.get("vertexSet", [])
                ori_labels = doc.get("labels", [])
                spo_list = []
                ent_index_to_name = {}
                for ent_idx, mentions in enumerate(vertex_set):
                    if len(mentions) == 0:
                        ent_name = f"Entity_{ent_idx}"
                    else:
                        first_mention = next((m for m in mentions if "coref" not in m), mentions[0])
                        name_val = first_mention.get("name", f"Entity_{ent_idx}")
                        if isinstance(name_val, list):
                            ent_name = " ".join(name_val).strip()
                        else:
                            ent_name = str(name_val).strip()
                        if not ent_name:
                            ent_name = f"Entity_{ent_idx}"
                    ent_index_to_name[ent_idx] = ent_name
                for label in ori_labels:
                    h_idx = label.get("h", -1)
                    t_idx = label.get("t", -1)
                    rel_val = label.get("r", "")

                    if h_idx not in ent_index_to_name or t_idx not in ent_index_to_name:
                        continue
                    if isinstance(rel_val, int):
                        rel_code = self.id2rel.get(rel_val, f"Unknown_Rel_{rel_val}")
                    else:
                        rel_code = str(rel_val).strip()

                    rel_name = self.rel_code2name.get(rel_code, rel_code)
                    spo = {
                        "subject": ent_index_to_name[h_idx],
                        "predicate": rel_name,
                        "object": ent_index_to_name[t_idx]
                    }
                    spo_list.append(spo)
                core_samples.append({
                    "text": text,
                    "spo_list": spo_list,
                    "doc_title": doc_title
                })
            except Exception as e:
                log.warning(f"{split} 数据集第 {doc_idx} 个文档提取失败: {str(e)}，跳过该文档")
                continue
        log.info(f"{split} 数据集核心字段提取完成：{len(core_samples)} 个有效样本")
        return core_samples

    def get_train_samples(self) -> List[Dict]:
        return self.train_samples

    def get_dev_samples(self) -> List[Dict]:
        return self.dev_samples

    def get_test_samples(self) -> List[Dict]:
        return self.test_samples

    def load_schema(self) -> None:
        if not self.fn_schema or not os.path.exists(self.fn_schema):
            log.warning(f"schema 文件路径无效或不存在（路径：{self.fn_schema}），尝试从 rel_code2name 生成默认 schema_dict")
            self._gen_default_schema_dict()
            return
        try:
            with open(self.fn_schema, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except Exception as e:
            log.error(f"schema 文件读取失败：{str(e)}，尝试生成默认 schema_dict")
            self._gen_default_schema_dict()
            return
        self.schema_dict = {}
        if isinstance(schema_data, list):
            if all(isinstance(item, str) for item in schema_data):
                for rel_name in schema_data:
                    rel_name = rel_name.strip()
                    if rel_name:  # 跳过空字符串
                        self.schema_dict[rel_name] = {
                            "predicate": rel_name,
                            "description": f"Auto-generated schema for: {rel_name}",
                            "source": "simple_list_json"
                        }
                log.info(f"从简单列表 JSON 加载 schema 成功，共 {len(self.schema_dict)} 个关系")
            elif all(isinstance(item, dict) and "predicate" in item for item in schema_data):
                for schema in schema_data:
                    pred_name = schema["predicate"].strip()
                    if pred_name:
                        self.schema_dict[pred_name] = schema
                log.info(f"从复杂对象列表 JSON 加载 schema 成功，共 {len(self.schema_dict)} 个关系")
            else:
                log.error("schema JSON 列表格式无效（元素需全为字符串或含 'predicate' 的字典），生成默认 schema_dict")
                self._gen_default_schema_dict()
        else:
            log.error("schema JSON 需为列表格式（如 [\"关系1\", \"关系2\"]），生成默认 schema_dict")
            self._gen_default_schema_dict()

    def _gen_default_schema_dict(self) -> None:
        unique_rel_names = set(self.rel_code2name.values())
        for rel_name in unique_rel_names:
            default_desc = f"Default schema for relation: {rel_name}"
            self.schema_dict[rel_name] = {
                "predicate": rel_name,
                "description": default_desc,
                "source": "auto_generated_from_rel_code2name"  # 标记为自动生成
            }
        log.info(f"生成默认 schema_dict 成功，共 {len(self.schema_dict)} 个关系：{list(self.schema_dict.keys())}")


    def get_relation_names(self) -> List[str]:
        return sorted(list(self.schema_dict.keys()))