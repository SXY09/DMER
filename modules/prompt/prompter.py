
from modules.tools import GetTaskDescription
from logging import getLogger
from DocRED import DocREDataModule
from .prompt_en import *
import json
from typing import List
from config.configurator import configs
from data_utils.data_handler_re import DataHandlerRE


class BasePormpter:
    def __init__(self, data_handler):
        self.data_handler: DocREDataModule = data_handler
        self.logger = getLogger('train_logger')
        self.language = "en"
        self.cfg = self.data_handler.cfg
        self.dataset_name = self.cfg.react_memory.data['name']  # 转大写统一格式
        self.logger.info(f"当前数据集：{self.dataset_name}，加载对应Prompt模板")
        self.supported_datasets = {"CDR", "GDA", "CHR", "BioRED"}
        if self.dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集：{self.dataset_name}，仅支持{self.supported_datasets}")

class PrompterReActMemory(BasePormpter):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "en":
            self.TEMPLATE_REFLEXION = CDR_TEMPLATE_REFLEXION_EN
            self.TEMPLATE_REACT = CDR_TEMPLATE_REACT_EN
            self.FIRST_STEP = CDR_FIRST_STEP_EN
            self.SECOND_STEP = CDR_SECOND_STEP_MEMORY_EN
            self.TEMPLATE_SUMMAY = CDR_TEMPLATE_SUMMAY_EN
            self.THIRD_STEP_REFLEXION_EN = CDR_THIRD_STEP_REFLEXION_EN
            self.FOURTH_STEP_ANALYZE_EN = CDR_FOURTH_STEP_ANALYZE_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        self.SUFFIX = SUFFIX

    def _get_suffix(self):
        suffix_map = {
            "CDR": CDR_SUFFIX,
            "GDA": GDA_SUFFIX,
            "CHR": CHR_SUFFIX,
        }
        return suffix_map[self.dataset_name]

    def get_react_prompt(self, text: str, tools_desc: str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)

    def get_react_first_step(self, task_description: str):
        return self.FIRST_STEP.format(task_description=task_description)

    def get_react_second_step(self, text: str, retrieved_examples: str):
        return self.SECOND_STEP.format(text=text, retrieved_examples=retrieved_examples)

    def get_react_third_step(self, text: str, retrieved_reflexion_examples: str):
        return self.THIRD_STEP_REFLEXION_EN.format(
            text=text, retrieved_reflexion_examples=retrieved_reflexion_examples
        )

    def get_react_suffix(self, pred_spo, golden_spo):
        return self.SUFFIX.format(pred_spo=pred_spo, golden_spo=golden_spo)

    def get_test_suffix(self, pred_spo):
        return self.FOURTH_STEP_ANALYZE_EN.format(pred_spo=pred_spo)

    def get_reflexion_prompt(self, text: str, golden: str, pred: str):
        golden, pred = json.dumps(golden, ensure_ascii=False), json.dumps(pred, ensure_ascii=False)
        return self.TEMPLATE_REFLEXION.format(text=text, golden=golden, pred=pred)

    def get_summary_prompt(self, text: str, golden: str, history: List[str]):
        if isinstance(golden, list):
            golden = json.dumps(golden, ensure_ascii=False)
        history = "\n".join(history)
        return self.TEMPLATE_SUMMAY.format(text=text, golden=golden, history=history)


