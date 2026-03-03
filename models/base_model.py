
from config.configurator import configs
from clients.openai_client import OpenAIClient
from DocRED import DocREDataModule
from modules.prompt.prompter import BasePormpter
import re, json
from logging import getLogger
from data_utils.data_handler_re import DataHandlerRE

class BaseModel:
    logger = getLogger('train_logger')
    data_handler: DocREDataModule# data handler
    data_handler1: DataHandlerRE
    prompter: BasePormpter
    # is_training = True

    def __init__(self, data_handler1: DataHandlerRE,data_handler=None):
        self.data_handler = data_handler
        self.data_handler1 = data_handler1
        if data_handler1 and hasattr(data_handler1, "cfg"):
            self.cfg = data_handler1.cfg
        else:
            self.cfg = None
            self.logger.warning("data_handler1 未传入或缺少 cfg 属性，将使用全局 configs 配置")
        self.llm = self.load_llm()

    def extract(self, text, idx):
        raise NotImplementedError

    def parse_llm_output(self, text: str) -> list[dict]:

        try:
            json_str = re.search(r'\{.*\}', text, flags=re.DOTALL).group()
            parsed = json.loads(json_str)
            spo_list = parsed['spo_list']
            return 0, spo_list
        except Exception as e:
            spo_list = []
            self.logger.error(
                f"[parse_llm_output失败] \n"
                f"输入文本: {text[:500]}...\n"
                f"错误类型: {type(e).__name__}\n"
                f"错误详情: {str(e)}"
            )
        return -1, spo_list

    def process_sample(self, sample, idx):
        ret = self.extract(sample['text'], idx)
        return {
            "spo_list_pred": ret["spo_list_pred"],
            "errorCode": ret.get("errorCode", 0),
            "history": ret.get("history", []),
            "final_output": ret.get("final_output", "")
        }

    def parse_output(self, llm_output: str) -> tuple[int, list[dict]]:

        raise NotImplementedError


    def train_sample(self, samle, idx):
        raise NotImplementedError

    def load_llm(self):
        llm_config = {}
        api_key = None


        if self.cfg and hasattr(self.cfg, "react_memory") and hasattr(self.cfg.react_memory, "llm"):
            react_llm_cfg = self.cfg.react_memory.llm
            llm_config['model_name'] = getattr(react_llm_cfg, "model_name", "")
            llm_config['temperature'] = getattr(react_llm_cfg, "temperature", 0)
            llm_config['max_tokens'] = getattr(react_llm_cfg, "max_tokens", 4096)
            api_key = getattr(react_llm_cfg, "api_key", "").strip()  # 读取 API 密钥
            self.logger.info("从 data_handler.cfg.react_memory.llm 加载 LLM 配置")


        if not llm_config.get("model_name") or not api_key:
            if "llm" in configs:
                global_llm_cfg = configs['llm']
                llm_config['model_name'] = global_llm_cfg.get('model_name', "gpt-4-turbo-preview")
                llm_config['temperature'] = global_llm_cfg.get('temperature', 0)
                llm_config['max_tokens'] = global_llm_cfg.get('max_tokens', 4096)
                api_key = global_llm_cfg.get('api_key', "").strip()  # 从全局 configs 读密钥
                self.logger.info("从全局 configs.llm 加载 LLM 配置（fallback）")
            else:
                raise ValueError("LLM 配置缺失！请在 YAML 或全局 configs 中配置 llm 相关参数")


        if not api_key:
            raise ValueError(
                "OpenAI API 密钥未配置！\n"
                "请在 YAML 中添加：react_memory.llm.api_key: '你的密钥'\n"
                "或在全局 configs 中添加：llm.api_key: '你的密钥'"
            )

        try:
            llm = OpenAIClient(
                api_key=api_key,
                model_name=llm_config['model_name'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
            )
            self.logger.info(f"LLM 初始化成功，模型名称: {llm_config['model_name']}")
            return llm
        except Exception as e:
            raise RuntimeError(f"LLM 初始化失败: {str(e)}") from e

    def query_llm(self, text, stop=None, temperature=None) -> str:
        res = self.llm.query_one(text, stop=stop, temperature=temperature)
        return res

    def log_prompt(self, prompt):
        self.logger.info(f"\n{'='*200}\nPrompt: {prompt}\n{'='*200}")

