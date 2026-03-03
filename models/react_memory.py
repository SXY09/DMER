
from config.configurator import configs
from models.base_model import BaseModel
import re, json
import importlib
from modules.tools import MEMORY_NAME2TOOL
from trainer.metrics_v2 import EvaluatorRE
from modules.memory.memory import CorrectMemory, BaseMemory, ReflexionMemory
from DocRED import DocREDataModule
from modules.module_utils import format_sample_str
from modules.prompt.prompter import PrompterReActMemory
from typing import Tuple, Optional, Any
import time

SUCCESS = 0
NO_RESULT_WITHIN_MAX_ITERATIONS = -1
NO_VALID_RESULT_WITHIN_MAX_RETRY = -2



class ReAct_Memory(BaseModel):
    mode: str = configs["model"]["mode"] if "mode" in configs["model"] else "dummy"
    stop: str = ["Output:", "Observation:"]        # LLM stop
    max_iterations: int = configs["model"]["max_iterations"]
    max_retry: int = configs["model"]["max_retry"]
    num_pre_history: int = configs["model"]["num_pre_history"]
    use_summary: bool = configs["model"]["use_summary"]
    debug: bool = configs["model"]["debug"]

    history: list = []              # for recording the history, CLEARED in each iteration
    tools: dict = {}                # list of tools
    memory_names: list = []         # list of memories
    prompter: PrompterReActMemory   # 

    # is_training = configs['train']['if_train']
    evaluator:EvaluatorRE = EvaluatorRE()

    def __init__(self, data_handler:DocREDataModule):
        super().__init__(data_handler)
        if configs['train']['if_predict'] or configs['train']['if_train']:
            self.init_memorys()
            self.init_tools()
        self.prompter = PrompterReActMemory(data_handler)

    def safe_query_llm(self, prompt, stop, temperature=0.5):
        for retry in range(self.max_retry):
            try:
                llm_output = self.query_llm(prompt, stop=stop, temperature=temperature)
                if llm_output is None:
                    raise ValueError("LLM returned empty response")
                if "403" in str(llm_output) and "flagged" in str(llm_output).lower():
                    self.logger.warning(f"API flagged (retry {retry + 1}), simplifying prompt...")
                    prompt = prompt[:int(
                        len(prompt) * 0.7)] + "\nPlease keep the response simple and follow the format strictly."
                    continue

                return llm_output.strip()  # 仅在成功时strip

            except Exception as e:
                self.logger.error(f"API call failed (retry {retry + 1}): {str(e)}")
                if retry == self.max_api_retry - 1:  # 最后一次重试失败
                    return None
        return None

    def init_tools(self):
        tools_activated = []
        for tool_name in configs['tools'].keys():
            if configs['tools'][tool_name]['open']:
                if tool_name in ["RetrieveExamples"]:
                    continue
                tools_activated.append(tool_name)
        for memory_name in self.memory_names:
            tools_activated.append(MEMORY_NAME2TOOL[memory_name])
        self.logger.info(f"Activated tools: {tools_activated}")

        module = importlib.import_module('modules.tools')
        for tool_name in tools_activated:
            tool = getattr(module, tool_name)(self.data_handler1, self.data_handler)
            self.tools[tool_name] = tool
        self.logger.info(f"Tools: {self.tools}")

    def init_memorys(self):
        if configs['memory']['CorrectMemory']['open']:
            self.memory_names.append('CorrectMemory')
            self.data_handler1.correct_memory = CorrectMemory()

            num_samples_init = configs['memory']['CorrectMemory']['num_samples_init']
            if num_samples_init > 0:
                self.logger.info(f"Init correct memory with {num_samples_init} samples.")
                samples_ds = self.data_handler1.ds_index.select(range(num_samples_init))
                samples_list = [samples_ds[i] for i in range(num_samples_init)]
                self.record_correct_memory(samples_list)
        if configs['memory']['ReflexionMemory']['open']:
            self.memory_names.append('ReflexionMemory')
            self.data_handler1.reflexion_memory = ReflexionMemory()

    def extract(self, text, idx, pred_spo=None, golden_spo=None,target_entity_pairs=None, is_test=False, reflection_prompt=None):
        debug = False

        text = json.dumps(text.strip(), ensure_ascii=False)
        if debug: self.logger.info(f"[idx={idx}] Input: {text}")
        history = []
        current_iter = 0

        if is_test:
            self_validation_context = (
                f"Observation: This is a test sample (no ground truth available). You need to self-verify your prediction:\n"
                f"1. Confirm there is text evidence (e.g., keywords like 'induce', 'risk') supporting the predicate.\n"
                f"2. Compare with similar examples in RetrieveCorrectMemory to ensure consistency.\n"
                f"3. If uncertain, mark the triple as 'needs verification' in your reflection."
            )
            history.append(self_validation_context)
        elif pred_spo is not None and golden_spo is not None:
            error_context = (
                f"Observation: Detected errors in the original prediction. Need to correct.\n"
                f"Original wrong prediction (pred_spo): {json.dumps(pred_spo, ensure_ascii=False)}\n"
                f"Ground truth (golden_spo): {json.dumps(golden_spo, ensure_ascii=False)}\n"
                f"Correction Rules: 1. Keep all entity pairs (subject & object) unchanged; 2. Fix 'predicate' to match golden_spo; 3. Do NOT add irrelevant triples."
            )
            history.append(error_context)
            if debug: self.logger.info(f"Added error context to history: {error_context[:200]}...")

        relation_names = self.data_handler1.get_relation_names()
        if target_entity_pairs and relation_names:
            relation_names_str = "|".join(relation_names)
            entity_constraint = (
                f"Observation: Only process relations for the following entity pairs; do NOT add other entity pairs:\n"
                f"{json.dumps(target_entity_pairs, ensure_ascii=False)}\n"
                f"Requirements:\n"
                f"1. Each entity pair must have exactly one relation output (no missing pairs).\n"
                f"2. The relation must be one of: `{relation_names_str}` (no other relation types allowed)."
            )
            history.append(entity_constraint)

        # ReAct-fashion
        # for _ in range(self.max_iterations):
        #     prompt = self.generate_prompt(
        #         text,
        #         pred_spo=pred_spo,
        #         golden_spo=golden_spo if not is_test else None,  # 测试时不传golden_spo
        #         target_entity_pairs=target_entity_pairs,
        #         is_test=is_test,
        #         reflection_prompt=reflection_prompt
        #     )
        for current_iter in range(self.max_iterations):
            prompt = self.generate_prompt(
                text,
                pred_spo=pred_spo,
                golden_spo=golden_spo if not is_test else None,
                target_entity_pairs=target_entity_pairs,
                is_test=is_test,
                reflection_prompt=reflection_prompt
            )
            prompt += f"\nNotice: You have {self.max_iterations - current_iter} iterations left. If no progress, call 'Finish' immediately."

            if idx < 5: self.log_prompt(prompt)
            for _ in range(self.max_retry):
                llm_output = self.safe_query_llm(prompt, stop=self.stop, temperature=0.2)    # can try different parameters
                err_code, parsed_res = self.parse_output(llm_output)
                if err_code == -1:
                    if debug: self.logger.error(f"error in parse_output: {llm_output}")
                    continue
                thought, action_name, args = parsed_res
                if action_name not in self.tools:
                    if debug: self.logger.error(f"error action_name: {action_name}. llm_output: {llm_output}")
                    continue
                if action_name == "Finish":
                    err_code, spo_list = self.parse_llm_output(args)
                    if err_code == -1:
                        if debug: self.logger.error(f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                        continue
                break
            else:
                self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
                return {
                    "spo_list_pred": [],
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
                }

            history.append(f"Thought: {thought}")
            if debug: self.logger.info(f"Thought: {thought}")
            if action_name == "Finish":
                err_code, spo_list = self.parse_llm_output(args)

                finish_output = json.dumps(args, ensure_ascii=False)
                history.append(f"Finish: {finish_output}")
                if debug: self.logger.info(f"Finish: {finish_output}")
                return {
                    "spo_list_pred": spo_list,
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": err_code,
                }
            else:
                observation = self.tools[action_name].call(args)
                history.append(f"Action: {action_name}({args})")
                history.append(f"Observation: {observation}")
                if debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
            return {
                "spo_list_pred": [],
                "history": history.copy(),
                "final_output": llm_output,
                "errorCode": NO_RESULT_WITHIN_MAX_ITERATIONS,
            }

    def train_sample(self, sample, idx):
        err_code = SUCCESS
        spo_list_pred = []
        summary_str = ""
        self.history = []

        text_str = json.dumps(sample['text'].strip(), ensure_ascii=False)
        if self.debug: self.logger.info(f"[idx={idx}] Input: {text_str}")

        for _ in range(self.max_iterations):
            # [0] prompt 
            prompt = self.generate_prompt(text_str)
            # if idx < 5: self.log_prompt(prompt)
            # Inner loop: try single action with max_retry limit
            err_code_, parsed_res = self.get_single_step(prompt)
            if err_code != 0:
                err_code = err_code_
                break
            thought, action_name, args = parsed_res

            # [1] thought
            self.history.append(f"Thought: {thought}")
            if self.debug: self.logger.info(f"Thought: {thought}")

            # [2] action
            if action_name == "Finish":
                err_code, spo_list_pred = self.parse_llm_output(json.loads(args))
                if err_code < 0:
                    self.logger.error(f"[ERROR] error in parse_llm_output: {args}")
                self.history.append(f"Finish: {args}")
                if self.debug: self.logger.info(f"Finish: {args}")
                # NEW: add refexion!
                f1 = self.get_eval_result(sample['spo_list'], spo_list_pred)
                if f1 < 1.0:
                    reflexion_text = self.get_reflexion(text_str, sample['spo_list'], spo_list_pred)
                    self.history.append(f"Reflexion: {reflexion_text}")
                    if self.debug: self.logger.info(f"Reflexion: {reflexion_text}")
                else:
                    pass
                break
            else:
                observation = self.tools[action_name].call(args)
                self.history.append(f"Action: {action_name}({args})")
                self.history.append(f"Observation: {observation}")
                if self.debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            err_code = NO_RESULT_WITHIN_MAX_ITERATIONS
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")

        self.record_correct_memory(sample)

        if self.use_summary:
            summary_str = self.get_summary(text_str, sample['spo_list'], self.history)
            self.history.append(f"Summary: {summary_str}")
            if self.debug: self.logger.info(f"Summary: {summary_str}")
        return {
            "spo_list_pred": spo_list_pred,
            "history": self.history.copy(),
            "summary": summary_str,
            "errorCode": err_code,
        }

    def get_single_step(self, prompt):
        for _ in range(self.max_retry):
            llm_output = self.query_llm(prompt, stop=self.stop, temperature=0.2)
            err_code, parsed_res = self.parse_output(llm_output)
            if err_code == -1:
                if self.debug: self.logger.error(f"error in parse_output: {llm_output}")
                continue
            thought, action_name, args = parsed_res
            if action_name not in self.tools:
                if self.debug: self.logger.error(f"error action_name: {action_name}. llm_output: {llm_output}")
                continue
            if action_name == "Finish":
                err_code, spo_list_pred = self.parse_llm_output(args)
                if err_code == -1:
                    if self.debug: self.logger.error(f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                    continue
            return 0, (thought, action_name, json.dumps(args, ensure_ascii=False))
        else:
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
            return NO_VALID_RESULT_WITHIN_MAX_RETRY, (None, None, None)

    def record_correct_memory(self, sample):
        if isinstance(sample, list):
            index_texts = [format_sample_str(s) for s in sample]
            self.data_handler1.correct_memory.add(index_texts)
        elif isinstance(sample, dict):
            index_text = format_sample_str(sample)
            self.data_handler1.correct_memory.add(index_text)
        else:
            raise Exception(f"Unknown sample type: {type(sample)}")
    # def record_correct_memory(self, sample):
    #     """优化：只记录高关联、高价值样本，控制内存上限1000个"""
    #     # 1. 仅处理dict类型（含text、spo_list、reflexion等字段）
    #     current_count = self.data_handler1.correct_memory.num_memory_items
    #     if current_count >= 1000:
    #         self.logger.info(f"CorrectMemory已达上限1000，不再添加新样本")
    #         return  # 满了，直接退出
    #
    #     if not isinstance(sample, dict) or "text" not in sample or "spo_list" not in sample:
    #         raise Exception(f"Sample must be dict with 'text' and 'spo_list': {type(sample)}")
    #
    #     # 2. 若有原始预测和golden，计算纠错价值（F1提升）
    #     f1_improve = 0.0
    #     if "pred_spo" in sample and "golden_spo" in sample:
    #         pred_f1 = self.get_eval_result(sample["golden_spo"], sample["pred_spo"])
    #         corrected_f1 = self.get_eval_result(sample["golden_spo"], sample["spo_list"])
    #         f1_improve = corrected_f1 - pred_f1
    #
    #     # 3. 识别错误模式（仅错误样本需要）
    #     error_pattern = "correct_sample"
    #     if "pred_spo" in sample and "golden_spo" in sample and f1_improve > 0:
    #         error_pattern, _ = self.get_error_pattern(sample["text"], sample["pred_spo"], sample["golden_spo"])
    #
    #     # 4. 筛选规则：满足以下任一条件才记录
    #     is_high_value = (
    #             f1_improve >= 0.1  # 纠错后F1提升显著
    #             or error_pattern in ["negation_misjudgment", "evidence_miss"]  # 核心错误模式
    #             or len(self.data_handler1.correct_memory) < 50  # 内存未满50个时宽松收录
    #     )
    #     if not is_high_value:
    #         self.logger.info(f"Skip low-value sample (pattern: {error_pattern}, f1_improve: {f1_improve:.3f})")
    #         return
    #
    #     # 5. 动态内存管理：上限100个，超出则淘汰低价值样本
    #     index_text = format_sample_str(sample)
    #     self.data_handler1.correct_memory.add(index_text)

        # 给样本添加元信息（用于淘汰）
        # sample["meta"] = {
        #     "error_pattern": error_pattern,
        #     "f1_improve": f1_improve,
        #     "record_time": time.time()
        # }
        # 若超出上限，淘汰“其他模式+F1提升最小”的样本
        # if len(self.data_handler1.correct_memory) > 500:
        #     # 假设correct_memory存储的是带meta的样本（需修改format_sample_str，保留meta）
        #     # 这里简化：按错误模式优先级+F1提升排序，删除最后1个
        #     sorted_samples = sorted(
        #         self.data_handler1.correct_memory.memory,  # 假设memory是样本列表
        #         key=lambda x: (
        #             x["meta"]["error_pattern"] not in ["negation_misjudgment", "evidence_miss"],
        #             x["meta"]["f1_improve"]
        #         )
        #     )
        #     self.data_handler1.correct_memory.memory = sorted_samples[:-1]  # 淘汰最后1个
        #     self.logger.info(
        #         f"CorrectMemory trimmed to 500 samples (removed low-value pattern: {sorted_samples[-1]['meta']['error_pattern']})")

    def get_reflexion(self, text, gloden, pred):
        # prompt = TEMPLATE_REFLEXION.format(text=text, golden=json.dumps(gloden, ensure_ascii=False), pred=json.dumps(pred, ensure_ascii=False))
        prompt = self.prompter.get_reflexion_prompt(text, gloden, pred)
        llm_output = self.query_llm(prompt, stop=self.stop, temperature=0).strip()
        if not llm_output:
            self.logger.error("Failed to generate reflexion (LLM returned empty)")
            return json.dumps({"error": "reflexion generation failed"}, ensure_ascii=False)
        reflexion = {
            "text": text,
            "golden": gloden,
            "pred": pred,
            "reflexion": llm_output,
        }
        return json.dumps(reflexion, ensure_ascii=False)

    def get_summary(self, text, golden, history):
        prompt = self.prompter.get_summary_prompt(text, golden, history)
        llm_output = self.query_llm(prompt, stop=self.stop, temperature=0).strip()
        if not llm_output:
            self.logger.error("Failed to generate summary (LLM returned empty)")
            return json.dumps({"error": "summary generation failed"}, ensure_ascii=False)
        return json.dumps(llm_output, ensure_ascii=False)

    def get_eval_result(self, golden, pred):
        self.evaluator.add(golden, pred)
        last_TP, last_FN, last_FP = self.evaluator.get_last_metric()
        f1 = round(last_TP / (last_TP + 0.5 * (last_FP + last_FN)), 4)
        # precision = round(last_TP / (last_TP + last_FP), 4)
        # recall = round(last_TP / (last_TP + last_FN), 4)
        # f1 = round(2 * precision * recall / (precision + recall), 4)
        return f1

    def generate_prompt(self, text, pred_spo=None, golden_spo=None,target_entity_pairs=None,is_test=False, reflection_prompt=None):
        tools_desc = "\n".join([f"- {tool.name}: {tool.get_description()}" for tool in self.tools.values()])
        task_description = self.tools['GetTaskDescription'].call()
        dataset_name = self.prompter.dataset_name
        dataset_config = {
            "CDR": {
                "example_spo1": {"subject": "gamma - vinyl GABA", "predicate": "chemical induced disease",
                                 "object": "visual field defects"},
                "example_spo2": {"subject": "cocaine", "predicate": "no chemical disease induction relation",
                                 "object": "substance abuse"},
                "evidence_keywords": "'induced', 'cause', 'lead to', 'result in'"
            },
            "GDA": {
                "example_spo1": {"subject": "BRCA1", "predicate": "gene disease association",
                                 "object": "breast cancer"},
                "example_spo2": {"subject": "TP53", "predicate": "no gene disease association",
                                 "object": "type 2 diabetes"},
                "evidence_keywords": "'associated with', 'linked to', 'predispose to', 'implicated in'"
            },
            "CHR": {
                "example_spo1": {"subject": "glucose", "predicate": "chemical metabolite interaction",
                                 "object": "pyruvate"},
                "example_spo2": {"subject": "insulin", "predicate": "no chemical metabolite interaction",
                                 "object": "cholesterol"},
                "evidence_keywords": "'catalyses', 'conversion', 'metabolic pathway', 'synthesis', 'reaction rate', 'substrate', 'product'"
            },
            "BIORED": {
                "example_spo1": {"subject": "TNF-α", "predicate": "biomedical association",
                                 "object": "inflammatory bowel disease"},
                "example_spo2": {"subject": "insulin", "predicate": "no biomedical association",
                                 "object": "Alzheimer's disease"},
                "evidence_keywords": "'biomedical interaction', 'bind to', 'regulate', 'express in'"
            }
        }
        current_config = dataset_config[dataset_name]

        if reflection_prompt is not None and reflection_prompt.strip():
            # 将反思约束添加到任务描述开头，确保 LLM 优先遵守
            task_description = f"{reflection_prompt}\n\n{task_description}"

        json_format_constraint = (
            "\n### Mandatory Output Format (Must strictly follow; invalid otherwise):\n"
            "When calling the `Finish` action, `ActionInput` must be a JSON dictionary with the `spo_list` key, structured as:\n"
            "{\n"
            '  "spo_list": [\n'
            '    {"subject": "Entity1", "predicate": "RelationType", "object": "Entity2"},\n'
            '    {"subject": "Entity3", "predicate": "RelationType", "object": "Entity4"},\n'
            "    ... （All triples must be included in the spo_list array）\n"
            "  ]\n"
            "}\n"
            "### Correct Example:\n"
            "{\n"
            f'  "spo_list": [\n'
            f'    {json.dumps(current_config["example_spo1"], ensure_ascii=False)},\n'
            f'    {json.dumps(current_config["example_spo2"], ensure_ascii=False)}\n'
            "  ]\n"
            "}"
            )

        if is_test:
            # 测试模式（无golden_spo）：强制要求使用两种Memory工具
            task_description += (
                "Objective: Extract relation triples from the text and self-verify their validity using TWO tools.\n"
                "Output format: {\"spo_list\": [{\"subject\": \"xxx\", \"predicate\": \"xxx\", \"object\": \"xxx\"}]}\n"
                "Self-Verification Rules:\n"
                "1. Predicate must be in the allowed list: {relation_names}\n"
                "2. **Must use tools for verification**:\n"
                "   - Use `RetrieveCorrectMemory` to align with correct labeling standards \n"
                "   - Use `RetrieveReflexionMemory` to avoid common mistakes \n"
                "3. Each triple must have explicit text evidence (e.g., 'associated with' or 'used for prevention of')."
            )
            task_description += json_format_constraint

        if pred_spo is not None and golden_spo is not None:
            task_description += (
                "\nCurrent Subtask: Correct errors in the original prediction to match ground truth.\n"
                "Correction Logic:\n"
                "1. **Entity Pair Strict Consistency**: The 'subject' and 'object' must be EXACTLY copied from `golden_spo` (including spaces, hyphens, and no extra modifiers). For example:\n"
                "   - If `golden_spo` has subject = \"gamma - vinyl GABA\", you MUST use this exact string (NOT \"Sub - chronic low dose gamma - vinyl GABA\" or other variants).\n"
                "   - Forbidden: Adding prefixes (e.g., \"Sub - chronic low dose\") or suffixes to entity names.\n"
                "2. **Only Fix Predicate**: Modify ONLY the 'predicate' field to match `golden_spo` (keep 'subject' and 'object' unchanged).\n"
                "3. **No Missing/Extra Triples**: Output all entity pairs from `golden_spo` (no omission, no adding irrelevant triples).\n"
                "\nNote:\n"
                "- Entity names are case-insensitive but character-sensitive (e.g., \"gamma - vinyl GABA\" ≠ \"Sub - chronic low dose gamma - vinyl GABA\").\n"
                "- If your output entity names do not match `golden_spo` exactly, the result will be considered invalid."
            )
            task_description += json_format_constraint


        retrieved_examples = self.tools['RetrieveCorrectMemory'].call(text)
        retrieved_reflexion_examples = self.tools['RetrieveReflexionMemory'].call(text)

        prompt = self.prompter.get_react_prompt(text, tools_desc) + \
            self.prompter.get_react_first_step(task_description) + \
            self.prompter.get_react_second_step(text, retrieved_examples) + \
            self.prompter.get_react_third_step(text, retrieved_reflexion_examples)
        if len(self.history) == 0:
            self.history.append(f"Action: GetTaskDescription()")
            self.history.append(f"Observation: {task_description}")
            self.history.append(f"Action: RetrieveCorrectMemory({text})")
            self.history.append(f"Observation: {retrieved_examples}")
            self.history.append(f"Action: RetrieveReflexionMemory({text})")
            self.history.append(f"Observation: {retrieved_reflexion_examples}")
            if pred_spo and golden_spo:
                self.history.append(
                    f"Observation: Original pred (truncated): {json.dumps(pred_spo, ensure_ascii=False)}...")
                self.history.append(
                    f"Observation: Ground truth (truncated): {json.dumps(golden_spo, ensure_ascii=False)}...")
            if pred_spo and is_test:
                self.history.append(
                    f"Observation: Original pred (truncated): {json.dumps(pred_spo, ensure_ascii=False)}...")
            # if self.debug:
            #     self.logger.info(f"Action: GetTaskDescription()")
            #     self.logger.info(f"Observation: {task_description}")
            #     self.logger.info(f"Action: RetrieveCorrectMemory({text})")
            #     self.logger.info(f"Observation: {retrieved_examples}")
        for history in self.history[self.num_pre_history * 2:]:             # Action+Observation
            prompt += history + "\n"
        if is_test:
            suffix = self.prompter.get_test_suffix(pred_spo=json.dumps(pred_spo, ensure_ascii=False)) + (
                "\nNow correct `pred_spo` using the following steps:\n"
                f"1. For each triple in `pred_spo`, check text evidence. \n"
                "2. Compare with `RetrieveCorrectMemory` examples. \n"
                "3. Avoid mistakes in `RetrieveReflexionMemory`.\n"
                "When calling 'Finish', wrap all triples in a JSON dictionary with 'spo_list' key. Example: {\"spo_list\": [ {...} ]}"
            )
        else:
            suffix = self.prompter.get_react_suffix(pred_spo=json.dumps(pred_spo, ensure_ascii=False),
                                                    golden_spo=json.dumps(golden_spo, ensure_ascii=False))+ (
            "\nWhen correcting, output the result as a JSON dictionary with 'spo_list' key (e.g., {\"spo_list\": [ {...} ]}). "
            "Do NOT output a raw list of triples."
        )
        format_example = f"""
            ### 正确输出示例：
            Thought: The predicate in triple 1 is wrong (should be '{current_config['example_spo2']['predicate']}').
            Action: RetrieveCorrectMemory
            ActionInput: "{current_config['example_spo2']['subject']} and {current_config['example_spo2']['object']}"
            Observation: [Retrieved examples showing no relation]

            Thought: All predicates are corrected.
            Action: Finish
            ActionInput: {json.dumps({'spo_list': [current_config['example_spo2']]}, ensure_ascii=False)}
            """
        prompt += format_example
        prompt += suffix
        #prompt += self.prompter.get_react_suffix(pred_spo=json.dumps(pred_spo, ensure_ascii=False),golden_spo=json.dumps(golden_spo, ensure_ascii=False))
        return prompt

    # def parse_output(self, llm_output: str):
    #     try:
    #         # 匹配最后一个 Action: 和对应的内容（忽略中间的其他 Action）
    #         regex = r".*Action:(.*?)\nActionInput:[\s]*(.*)$"
    #         match = re.search(regex, llm_output, re.DOTALL)
    #         if not match:
    #             return -1, None
    #         # 提取 Thought（最后一个 Action 之前的内容）
    #         thought_part = re.sub(r"Action:.*$", "", llm_output, flags=re.DOTALL).strip()
    #         action = match.group(1).strip()
    #         args = match.group(2).strip()
    #         thought = json.dumps(thought_part, ensure_ascii=False)
    #         return 0, (thought, action, args)
    #     except Exception as e:
    #         return -1, None

    def parse_output(self, llm_output: str):
        try:
            regex = r".*Action:(.*?)\nActionInput:[\s]*(.*)$"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                return -1, None

            thought_part = re.sub(r"Action:.*$", "", llm_output, flags=re.DOTALL).strip()
            thought = json.dumps(thought_part, ensure_ascii=False)

            action = match.group(1).strip()
            args = match.group(2).strip()

            args = re.sub(r"^```(json)?", "", args)
            args = re.sub(r"```$", "", args).strip()

            if args.startswith("{"):
                brace_count = 0
                end_idx = len(args)
                for i, c in enumerate(args):
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                args = args[:end_idx]
            elif args.startswith("["):
                bracket_count = 0
                end_idx = len(args)
                for i, c in enumerate(args):
                    if c == "[":
                        bracket_count += 1
                    elif c == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                args = args[:end_idx]

            json.loads(args)

            return 0, (thought, action, args)
        except json.JSONDecodeError:
            return -1, None
        except Exception as e:
            return -1, None

    # def parse_output(self, llm_output: str):
    #     try:
    #         # 1. 清理大模型输出（去除多余空格、换行）
    #         clean_output = re.sub(r"\s+", " ", llm_output.strip())  # 多个空格/换行替换为单个空格
    #         # 2. 容错性正则：匹配Action和ActionInput（忽略大小写、空格、括号）
    #         # 支持格式：Action: RetrieveCorrectMemory ActionInput: "text" 或 Action: RetrieveCorrectMemory() ActionInput: text
    #         regex = r".*Action:\s*([A-Za-z0-9_]+)(?:\(\))?\s*ActionInput:\s*(.*?)$"
    #         match = re.search(regex, clean_output, re.DOTALL | re.IGNORECASE)
    #         if not match:
    #             self.logger.error(f"解析失败：未匹配到Action/ActionInput，原始输出：{clean_output[:100]}")
    #             return -1, None
    #
    #         # 3. 提取并验证参数
    #         action = match.group(1).strip()
    #         args = match.group(2).strip()
    #         # 处理args可能的JSON格式（如带引号、转义符）
    #         try:
    #             args = json.loads(args)  # 尝试解析为JSON（若大模型输出JSON格式的args）
    #         except json.JSONDecodeError:
    #             # 若不是JSON，直接作为字符串使用（如纯文本）
    #             pass
    #
    #         # 4. 提取Thought（最后一个Action之前的内容）
    #         thought_part = re.sub(regex, "", clean_output, flags=re.DOTALL | re.IGNORECASE).strip()
    #         if not thought_part:
    #             thought_part = "No explicit thought provided."  # 兜底，避免空值
    #         thought = json.dumps(thought_part, ensure_ascii=False)
    #
    #         return 0, (thought, action, args)
    #     except Exception as e:
    #         self.logger.error(f"解析异常：{str(e)}，原始输出：{llm_output[:100]}")
    #         return -1, None

    # def parse_output(self, llm_output: str) -> Tuple[int, Optional[Tuple[str, str, Any]]]:
    #     try:
    #         # 1. 保留原始输出中的换行（便于匹配多行格式），仅清理首尾空格
    #         clean_output = llm_output.strip()
    #         # 2. 容错性正则：匹配Thought、Action、ActionInput（支持多行和格式变体）
    #         # 支持格式：
    #         # - Thought: ...\nAction: ...\nActionInput: ...
    #         # - Thought: ... Action: ... ActionInput: ...
    #         regex = r"Thought:\s*(.*?)\s*Action:\s*([A-Za-z0-9_]+)(?:\(\))?\s*ActionInput:\s*(.*)$"
    #         match = re.search(regex, clean_output, re.DOTALL | re.IGNORECASE)
    #         if not match:
    #             self.logger.error(f"解析失败：未匹配到Action/ActionInput，原始输出：{clean_output[:200]}")
    #             return -1, None
    #
    #         # 3. 提取基础字段
    #         thought_part = match.group(1).strip()
    #         action = match.group(2).strip()
    #         raw_args = match.group(3).strip()  # 原始ActionInput内容（可能含多余文本）
    #
    #         # 4. 增强型JSON解析：处理ActionInput中的JSON（尤其是Finish动作）
    #         parsed_args = raw_args  # 默认为原始字符串
    #         if action == "Finish":
    #             # 4.1 提取原始args中的JSON结构（匹配最外层{}，支持多行）
    #             json_pattern = re.compile(r"\{[\s\S]*?\}", re.DOTALL)  # 匹配从{到}的所有内容
    #             json_matches = json_pattern.findall(raw_args)
    #             if json_matches:
    #                 # 取最长匹配（通常是完整JSON）
    #                 candidate_json = max(json_matches, key=len)
    #                 # 4.2 清理JSON中的干扰字符
    #                 candidate_json = candidate_json.strip()
    #                 candidate_json = re.sub(r"[;,]\s*$", "", candidate_json)  # 去除末尾分号/逗号
    #                 candidate_json = re.sub(r"//.*?\n", "", candidate_json)  # 去除单行注释
    #                 candidate_json = re.sub(r"/\*.*?\*/", "", candidate_json, flags=re.DOTALL)  # 去除多行注释
    #                 # 4.3 修复常见JSON格式错误（LLM易犯的笔误）
    #                 candidate_json = re.sub(r"(['\"])\s*:\s*([^'\"]\w+)", r"\1: \"\2\"", candidate_json)  # 补全值的引号
    #                 candidate_json = re.sub(r",\s*}", "}", candidate_json)  # 去除尾逗号
    #                 # 4.4 尝试解析修复后的JSON
    #                 try:
    #                     parsed_args = json.loads(candidate_json)
    #                 except json.JSONDecodeError as e:
    #                     self.logger.warning(
    #                         f"Finish动作JSON修复后仍解析失败：{str(e)}，原始JSON候选：{candidate_json[:200]}"
    #                     )
    #                     # 保留原始字符串作为 fallback
    #         else:
    #             # 非Finish动作：尝试解析为JSON（如工具参数是结构化数据）
    #             try:
    #                 parsed_args = json.loads(raw_args)
    #             except json.JSONDecodeError:
    #                 # 非JSON格式则保持字符串（如检索文本）
    #                 pass
    #
    #         # 5. 处理Thought（兜底空值）
    #         if not thought_part:
    #             thought_part = "No explicit thought provided."
    #         thought = json.dumps(thought_part, ensure_ascii=False)
    #
    #         return 0, (thought, action, parsed_args)
    #
    #     except Exception as e:
    #         self.logger.error(f"解析异常：{str(e)}，原始输出：{llm_output[:200]}")
    #         return -1, None

    def _clean_args(self, args_raw, action):
        if not args_raw:
            return None

        args_clean = re.sub(r"^```(json)?", "", args_raw)
        args_clean = re.sub(r"```$", "", args_clean).strip()
        args_clean = re.sub(r"//.*$", "", args_clean, flags=re.MULTILINE)  # 移除单行注释

        if action == "Finish":
            if args_clean.startswith("{") and "}" in args_clean:
                args_clean = args_clean[:args_clean.rfind("}") + 1]
            args_clean = re.sub(r",\s*}", "}", args_clean)
            try:
                json.loads(args_clean)
                return args_clean
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON in Finish: {args_clean[:100]}")
                return None

        return args_clean

    def get_error_pattern(self, text, pred_spo, golden_spo):
        dataset_name = self.prompter.dataset_name
        neg_keywords = ["not", "no", "unrelated", "without"]  # 否定词列表
        text_lower = text.lower()

        pred_relations = [spo["predicate"] for spo in pred_spo]
        golden_relations = [spo["predicate"] for spo in golden_spo]

        has_neg = any(kw in text_lower for kw in neg_keywords)
        if has_neg and "no " not in pred_relations[0] and "no " in golden_relations[0]:
            return "negation_misjudgment", neg_keywords

        pred_entity_pairs = [(spo["subject"].lower(), spo["object"].lower()) for spo in pred_spo]
        golden_entity_pairs = [(spo["subject"].lower(), spo["object"].lower()) for spo in golden_spo]
        if pred_entity_pairs != golden_entity_pairs and "no " in pred_relations[0]:
            return "entity_alias", []

        if len(text) > 500 and "\n" in text and "no " in pred_relations[0] and "no " not in golden_relations[0]:
            return "long_distance", []

        return "other", []




