import os
import shutil

import math
import json
import argparse
from collections import defaultdict
from copy import deepcopy
import logging

#import rich
import torch
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from utils import set_seed,assign_distance_bucket
import hydra
from utils import get_lr, print_config_tree

from data_utils.data_handler_re import DataHandlerRE
from models.react_memory import ReAct_Memory, SUCCESS, NO_RESULT_WITHIN_MAX_ITERATIONS, NO_VALID_RESULT_WITHIN_MAX_RETRY
from trainer.metrics_v2 import EvaluatorRE
import re
import pickle

log = logging.getLogger(__name__)


def parse_docre_pred_to_spo_simple(pred_labels, batch, sample_idx):
    docre_spo = []
    REL_ID_TO_NAME = {
        1: "chemical induced disease",
        0: "no chemical disease induction relation"
    }

    try:
        original_spo_list = batch["spo_lists"][sample_idx]  # 原始三元组（含 subject/object）
        sample_text = batch["texts"][sample_idx] if sample_idx < len(batch["texts"]) else ""
        sample_title = batch["titles"][sample_idx] if sample_idx < len(batch["titles"]) else f"doc_{sample_idx}"

        align_len = min(len(pred_labels), len(original_spo_list))
        for idx in range(align_len):

            pred = pred_labels[idx]
            pred_rel_id = np.argmax(pred)


            pred_rel_name = REL_ID_TO_NAME.get(pred_rel_id, f"unknown_relation_{pred_rel_id}")


            original_spo = original_spo_list[idx]
            predicted_spo = {
                "subject": original_spo["subject"],
                "predicate": pred_rel_name,
                "object": original_spo["object"]
            }
            docre_spo.append(predicted_spo)


        if len(docre_spo) < len(original_spo_list):
            docre_spo += original_spo_list[len(docre_spo):]


        sample_golden = {
            "text": sample_text,
            "golden_spo": original_spo_list,
            "doc_title": sample_title,
            "sample_idx": sample_idx
        }

        return docre_spo, sample_golden

    except Exception as e:
        log.error(f"处理样本 {sample_idx} 时出错：{str(e)}", exc_info=True)

        return [], {"text": "", "golden_spo": [], "doc_title": "", "sample_idx": sample_idx}

def parse_docre_pred_to_spo(pred_labels, sub_sample):
    hts = sub_sample["hts"]
    entities = sub_sample["entities"]
    id2rel = sub_sample.get("id2rel", {})
    spo_list = []

    for i, (h_idx, t_idx) in enumerate(hts):
        if i >= len(pred_labels):
            continue
        rel_id = pred_labels[i][0]
        has_rel = pred_labels[i][1]
        if has_rel == 1:
            rel_name = id2rel.get(rel_id, f"relation_{rel_id}")

            if 0 <= h_idx < len(entities) and 0 <= t_idx < len(entities):
                subject = entities[h_idx]["name"].strip()
                object_ = entities[t_idx]["name"].strip()
                if subject and object_:
                    spo_list.append({
                        "subject": subject,
                        "predicate": rel_name,
                        "object": object_
                    })
    return spo_list

def get_entity_pair_key(subject, object_, normalize=True):
    if normalize:
        subject = subject.strip()
        object_ = object_.strip()
    return (subject, object_)


def spo_to_docre_pred(corrected_spo, docre_spo, original_pred, batch, sample_idx):
    REL_NAME_TO_ID = {
        "gene disease association": 1,
        "no gene disease association": 0,
        "chemical induced disease": 1,
        "no chemical disease induction relation": 0,
        "unknown_0": 0,
        "unknown_1": 1
    }

    entity_pair_to_index = {}
    for idx, spo in enumerate(docre_spo):
        subj = spo["subject"].strip().lower()
        obj = spo["object"].strip().lower()
        entity_pair_to_index[(subj, obj)] = idx

    num_pairs = len(docre_spo)

    final_pred = np.zeros((num_pairs, 2), dtype=np.float32)


    if original_pred is not None and isinstance(original_pred, np.ndarray) and original_pred.size > 0:

        original_labels = np.argmax(original_pred, axis=1)  # 形状：(n,)

        for i in range(min(len(original_labels), num_pairs)):
            final_pred[i][original_labels[i]] = 1.0  # 生成one-hot


    HIGH_CONFIDENCE_THRESHOLD = 0.8
    for corrected in corrected_spo:

        corrected_subj = corrected["subject"].strip().lower()
        corrected_obj = corrected["object"].strip().lower()
        corrected_key = (corrected_subj, corrected_obj)

        if corrected_key not in entity_pair_to_index:
            continue
        idx = entity_pair_to_index[corrected_key]
        if idx >= num_pairs:
            continue


        corrected_predicate = corrected.get("predicate", "")
        if corrected_predicate not in REL_NAME_TO_ID:
            continue
        corrected_rel_id = REL_NAME_TO_ID[corrected_predicate]


        original_confidence = 0.0
        if original_pred is not None and isinstance(original_pred, np.ndarray) and original_pred.size > 0:
            if idx < original_pred.shape[0]:
                original_confidence = np.max(original_pred[idx])  # 原始概率的最大值


        if original_confidence < HIGH_CONFIDENCE_THRESHOLD:

            final_pred[idx] = np.zeros(2, dtype=np.float32)
            final_pred[idx][corrected_rel_id] = 1.0

    return final_pred


def train(cfg,data_handler,datamodule, model, react_memory):
    args = cfg.train
    if args.seed:  #  
        set_seed(args.seed)
    model.to(args.device)

    MEMORY_MAX_SIZE = 500

    train_dataset, train_dataloader = datamodule.train_dataset, datamodule.train_dataloader()
    dev_dataset, dev_dataloader = datamodule.dev_dataset, datamodule.dev_dataloader()
    test_dataset, test_dataloader = datamodule.test_dataset, datamodule.test_dataloader()

    train_samples = data_handler.get_train_samples()
    id2rel = data_handler.id2rel


    total_steps = args.epochs * (len(train_dataloader) // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps} = {args.epochs} epoch * ({len(train_dataloader)} batch // {args.gradient_accumulation_steps})")
    print(f"Warmup steps: {warmup_steps} = {total_steps} total steps * {args.warmup_ratio} warmup ratio")

    new_layer = ["extractor", "projection", "classifier", "conv", "graph_conv", "mu_encoder"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
         "lr": args.classifier_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = amp.GradScaler()

    num_steps = 0
    dev_best_score = -1
    test_best_score = -1
    model_name_or_path = cfg.model.model_name_or_path
    model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
    for epoch in range(args.epochs):

        is_react_epoch = (epoch >= 40) and (epoch % 2 == 0)

        print("epoch: " + str(epoch))
        optimizer.zero_grad()
        for step, (batch, sample) in enumerate(tqdm(zip(train_dataloader, train_samples))):
            model.train()
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'hts': batch['hts'],
                'sent_pos': batch['sent_pos'],
                'entity_pos': batch['entity_pos'],
                'mention_pos': batch['mention_pos'],
                'entity_types': batch['entity_types'],
                'men_graphs': batch['men_graphs'].to(args.device),
                'labels': batch['labels'],
            }

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(**inputs)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                num_steps += 1
            if (args.log_steps > 0 and step % args.log_steps == 0) or (step + 1 == len(train_dataloader)):
                print(f"{epoch}/{step}/{len(train_dataloader)}: current loss {round(loss.item(), 4)}")

            if cfg.react_memory.enable and react_memory is not None and (
                    step + 1) % args.gradient_accumulation_steps == 0 and is_react_epoch:

                with torch.no_grad():
                    inputs["labels"] = None
                    pred_labels = model(**inputs)
                    pred_labels = pred_labels.cpu().numpy()

                sample_pred_len = [len(spos) for spos in batch["spo_lists"]]  # 每个样本的 spo 数量
                pred_labels_split = []
                start = 0
                for len_ in sample_pred_len:
                    pred_labels_split.append(pred_labels[start:start + len_])
                    start += len_

                for sample_idx in range(len(batch["spo_lists"])):
                    docre_spo, sample_golden = parse_docre_pred_to_spo_simple(
                        pred_labels=pred_labels_split[sample_idx],  # 单个样本的预测结果
                        batch=batch,
                        sample_idx=sample_idx
                    )

                    if not sample_golden["golden_spo"] or not sample_golden["text"]:
                        continue

                    golden_spo = sample_golden["golden_spo"]
                    react_evaluator = EvaluatorRE()
                    react_evaluator.add(golden_spo, docre_spo)
                    metric_dict = react_evaluator.get_metric_dict()
                    f1 = metric_dict.get("f1", 0.0)
                    correct_sample1 = {
                        "text": batch["texts"][sample_idx],
                        "spo_list": batch["spo_lists"][sample_idx],
                        "title": batch["titles"][sample_idx],
                    }
                    current_count = react_memory.data_handler1.correct_memory.num_memory_items
                    if current_count >= MEMORY_MAX_SIZE:
                        continue

                    if f1 >= 1.0:
                        react_memory.record_correct_memory(correct_sample1)
                    else:
                        react_result = react_memory.extract(correct_sample1["text"], idx=batch["titles"][sample_idx],pred_spo=docre_spo,golden_spo=correct_sample1["spo_list"])
                        if react_result["errorCode"] == SUCCESS:
                            corrected_spo = react_result["spo_list_pred"]
                            if corrected_spo:
                                corrected_sample = deepcopy(correct_sample1)
                                corrected_sample["spo_list"] = corrected_spo
                                reflexion_text = react_memory.get_reflexion(
                                    text=correct_sample1["text"],
                                    gloden=correct_sample1["spo_list"],
                                    pred=docre_spo
                                )
                                react_memory.data_handler1.reflexion_memory.add(reflexion_text)

                                summary_text = react_memory.get_summary(
                                    text=correct_sample1["text"],
                                    golden=correct_sample1["spo_list"],
                                    history=react_result["history"]  # 从react_result获取处理历史
                                )

                                corrected_sample["reflexion"] = reflexion_text
                                corrected_sample["summary"] = summary_text
                                corrected_sample["final_output"] = react_result["final_output"]
                                corrected_sample["errorCode"] = react_result["errorCode"]
                                corrected_sample["history"] = react_result["history"]
                                corrected_sample["golden_spo"] = correct_sample1["spo_list"]
                                react_memory.record_correct_memory(corrected_sample)

        if (step + 1) == len(train_dataloader) \
                    or (args.evaluation_steps > 0
                        and num_steps > total_steps // 2
                        and num_steps % args.evaluation_steps == 0
                        and step % args.gradient_accumulation_steps == 0
                        and num_steps > args.start_steps):
                dev_score, dev_output = evaluate(cfg, model, dev_dataset, dev_dataloader,react_memory, epoch,tag="dev")
                test_score, test_output = evaluate(cfg, model, test_dataset, test_dataloader,react_memory,epoch,tag="test")
                print(dev_output)
                print(test_output)

                lm_lr, classifier_lr = get_lr(optimizer)
                print(f'Current Step: {num_steps}, Current PLM lr: {lm_lr}, Current Classifier lr: {classifier_lr}')

                if dev_score > dev_best_score or dev_score > 60:
                    dev_best_score = dev_score

                    if test_score > test_best_score:
                        test_best_score = test_score
                        save_dir = args.save_best_path
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        pre_max_model = [saved_model_name for saved_model_name in os.listdir(save_dir) if
                                         saved_model_name[:saved_model_name.find('_')] == model_name_or_path]
                        if len(pre_max_model) == 0:
                            pre_max_score = -1
                        else:
                            pre_max_score = max(float(saved_model_name[saved_model_name.rfind('_') + 1:])
                                                for saved_model_name in pre_max_model)
                        if args.save_best_path and test_score > pre_max_score:
                            sub_save_dir = f"{save_dir}/{model_name_or_path}_{round(test_score, 2)}"
                            save_model_path = f"{sub_save_dir}/cdr_model.pth"
                            save_config_path = f"{sub_save_dir}/config.txt"
                            if not os.path.exists(sub_save_dir):
                               os.makedirs(sub_save_dir)
                            torch.save(model.state_dict(), save_model_path)
                            print_config_tree(cfg, open(save_config_path, "w"))
                            if pre_max_score != -1:
                               shutil.rmtree(f"{save_dir}/{model_name_or_path}_{pre_max_score}")
                if args.save_last_path:

                    save_last_dir = os.path.dirname(args.save_last_path)

                    if not os.path.exists(save_last_dir):
                        os.makedirs(save_last_dir, exist_ok=True)

                    torch.save(model.state_dict(), args.save_last_path)

    if cfg.react_memory.enable and react_memory is not None:
        save_memory_if_not_exist(data_handler, cfg)


def evaluate(cfg, model, dataset, dataloader,react_memory, epoch,tag="dev"):
    assert tag in {"dev", "test"}
    args = cfg.train

    if tag == "dev":
        print("Evaluating")
    else:
        print("Testing")
    preds, golds, dists, ent_dis ,preds_origin= [], [], [], [], []
    id2rel = dataset.id2rel

    is_react_epoch = False
    if epoch is not None:

        is_react_epoch = (epoch >= 40) and (epoch % 2 == 0)

        if tag == "dev":
            is_react_epoch = False
    print(f"当前评估是否启用 ReAct 反思：{is_react_epoch}")


    model.to(args.device)
    for batch in dataloader:
        model.eval()

        inputs = {
            'input_ids': batch['input_ids'].to(args.device),
            'attention_mask': batch['attention_mask'].to(args.device),
            'hts': batch['hts'],
            'sent_pos': batch['sent_pos'],
            'entity_pos': batch['entity_pos'],
            'mention_pos': batch['mention_pos'],
            'entity_types': batch['entity_types'],
            'men_graphs': batch['men_graphs'].to(args.device),
            'labels': None,

        }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()

            if cfg.react_memory.enable and react_memory is not None and is_react_epoch:
                batch_pred = []
                sample_spo_counts = [len(spo_list) for spo_list in batch["spo_lists"]]
                num_samples = len(sample_spo_counts)
                split_indices = [0]
                current_idx = 0
                for count in sample_spo_counts:
                    current_idx += count
                    split_indices.append(current_idx)
                for sample_idx in range(num_samples):

                    start = split_indices[sample_idx]
                    end = split_indices[sample_idx + 1]
                    sample_pred_labels = pred[start:end]

                    assert len(sample_pred_labels) == sample_spo_counts[sample_idx], \
                        f"样本{sample_idx}：预测长度{len(sample_pred_labels)} != spo数量{sample_spo_counts[sample_idx]}"

                    docre_spo, sample_golden = parse_docre_pred_to_spo_simple(
                        sample_pred_labels, batch, sample_idx
                    )
                    try:
                        react_result = react_memory.extract(
                            text=sample_golden["text"],
                            idx=sample_golden["doc_title"],
                            pred_spo=docre_spo,
                            is_test=True
                        )
                        corrected_spo = react_result["spo_list_pred"] if react_result[
                                                                             "errorCode"] == SUCCESS else docre_spo
                    except Exception as e:
                        print(f"ReAct error (sample {sample_idx}): {str(e)[:50]}")
                        corrected_spo = docre_spo

                    corrected_pred = spo_to_docre_pred(corrected_spo, docre_spo, sample_pred_labels, batch, sample_idx)
                    batch_pred.append(corrected_pred)

                pred[np.isnan(pred)] = 0
                preds_origin.append(pred)
                pred = np.concatenate(batch_pred, axis=0)
                pred[np.isnan(pred)] = 0
                preds.append(pred)
                golds.append(np.concatenate([np.array(label, np.float32) for label in batch['labels']], axis=0))
                dists.append(np.concatenate([np.array(dist, np.float32) for dist in batch['dists']], axis=0))
                ent_dis.append(np.concatenate([np.array(dist, np.float32) for dist in batch['ent_dis']], axis=0))
            else:
                pred[np.isnan(pred)] = 0
                preds.append(pred)
                golds.append(np.concatenate([np.array(label, np.float32) for label in batch['labels']], axis=0))
                dists.append(np.concatenate([np.array(dist, np.float32) for dist in batch['dists']], axis=0))
                ent_dis.append(np.concatenate([np.array(dist, np.float32) for dist in batch['ent_dis']], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    dists = np.concatenate(dists, axis=0).astype(np.float32)
    ent_dis = np.concatenate(ent_dis, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    fn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    tn = ((preds[:, 1] == 0) & (golds[:, 1] == 0)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    fer = tn / (tn + fp + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    tp_intra = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 0)).astype(np.float32).sum()
    fn_intra = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    fp_intra = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    precision_intra = tp_intra / (tp_intra + fp_intra + 1e-5)
    recall_intra = tp_intra / (tp_intra + fn_intra + 1e-5)
    f1_intra = 2 * precision_intra * recall_intra / (precision_intra + recall_intra + 1e-5)

    tp_inter = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 1)).astype(np.float32).sum()
    fn_inter = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    fp_inter = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    precision_inter = tp_inter / (tp_inter + fp_inter + 1e-5)
    recall_inter = tp_inter / (tp_inter + fn_inter + 1e-5)
    f1_inter = 2 * precision_inter * recall_inter / (precision_inter + recall_inter + 1e-5)

    distance_buckets = [8, 32, 64, 128]
    bucket_labels = np.array([assign_distance_bucket(d, distance_buckets) for d in ent_dis])

    buckets = len(distance_buckets) + 1
    dis_tp = np.zeros(buckets, dtype=np.float32)
    dis_fp = np.zeros(buckets, dtype=np.float32)
    dis_fn = np.zeros(buckets, dtype=np.float32)

    for pred, gold, bucket in zip(preds[:, 1], golds[:, 1], bucket_labels):
        if pred == gold == 1:
            dis_tp[bucket] += 1
        elif pred == 1 and gold != 1:
            dis_fp[bucket] += 1
        elif pred != 1 and gold == 1:
            dis_fn[bucket] += 1
    dis_precision = dis_tp / (dis_tp + dis_fp + 1e-5)
    dis_recall = dis_tp / (dis_tp + dis_fn + 1e-5)
    dis_f1_scores = 2 * dis_precision * dis_recall / (dis_precision + dis_recall + 1e-5)  # 防止分母为0

    output = {
        "{}_p".format(tag): precision * 100,
        "{}_r".format(tag): recall * 100,
        "{}_fer".format(tag): fer * 100,
        "{}_f1".format(tag): f1 * 100,
        "{}_f1_intra".format(tag): f1_intra * 100,
        "{}_f1_inter".format(tag): f1_inter * 100,
        "{}_f1_1".format(tag): dis_f1_scores[0] * 100,
        "{}_f1_2".format(tag): dis_f1_scores[1] * 100,
        "{}_f1_3".format(tag): dis_f1_scores[2] * 100,
        "{}_f1_4".format(tag): dis_f1_scores[3] * 100,
        "{}_f1_5".format(tag): dis_f1_scores[4] * 100,
    }
    return f1, output

def save_memory_if_not_exist(data_handler, cfg):

    memory_dir = get_memory_save_dir(cfg)
    correct_mem_path = os.path.join(memory_dir, "correct_memory.pkl")
    reflex_mem_path = os.path.join(memory_dir, "reflexion_memory.pkl")

    if os.path.exists(correct_mem_path) and os.path.exists(reflex_mem_path):
        log.info(f" 记忆已存在（{memory_dir}），跳过存储")
        return
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
    with open(correct_mem_path, "wb") as f:
        pickle.dump(data_handler.correct_memory, f)
    with open(reflex_mem_path, "wb") as f:
        pickle.dump(data_handler.reflexion_memory, f)
    log.info(f" 记忆已保存到 {memory_dir}")


def load_memory_from_cfg(data_handler, cfg):

    memory_dir = get_memory_save_dir(cfg)
    correct_mem_path = os.path.join(memory_dir, "correct_memory.pkl")
    reflex_mem_path = os.path.join(memory_dir, "reflexion_memory.pkl")

    if not os.path.exists(correct_mem_path) or not os.path.exists(reflex_mem_path):
        log.warning(f" 记忆目录不存在（{memory_dir}），将重新生成记忆")
        return False
    with open(correct_mem_path, "rb") as f:
        data_handler.correct_memory = pickle.load(f)
    with open(reflex_mem_path, "rb") as f:
        data_handler.reflexion_memory = pickle.load(f)
    log.info(f" 从 {memory_dir} 加载记忆成功")
    return True


def get_memory_save_dir(cfg):

    dataset_name = cfg.react_memory.data.name
    llm_model_name = cfg.react_memory.llm.model_name

    safe_llm_name = re.sub(r"[\/:]", "-", llm_model_name)
    base_save_dir = cfg.train.save_memory
    memory_dir = os.path.join(base_save_dir, f"react_memory_{dataset_name}_{safe_llm_name}")
    return memory_dir


def generate_reflection_prompt(text, original_pred, original_confidence, entity_pairs, threshold=0.8):

    processed_pred = []
    for pred in original_pred:

        if isinstance(pred, (list, np.ndarray)) and len(pred) == 2:
            processed_pred.append(np.argmax(pred))

        elif isinstance(pred, (int, float)) and (pred == 0 or pred == 1):
            processed_pred.append(int(pred))

        else:
            print(f"警告：无效的预测格式 {pred}，默认按 0 处理")
            processed_pred.append(0)
    original_pred = processed_pred

    low_confidence_pairs = []
    for i, (h, t) in enumerate(entity_pairs):
        if original_confidence[i] < threshold:
            #
            pred_relation = "chemical induced disease" if original_pred[
                                                              i] == 1 else "no chemical disease induction relation"
            low_confidence_pairs.append(
                f"Entity Pair {i + 1}: {h} and {t}, Model Prediction: {pred_relation}, Confidence Score: {original_confidence[i]:.2f}"
            )

    if not low_confidence_pairs:
        return "All predictions have a confidence score ≥ 0.8; no modification is needed."

    evidence_rules = """
    ### Mandatory Evidence Rules:
    1. To predict "chemical induced disease" (Class 1), the text MUST contain explicit causal keywords:
       - Causal terms: "induce", "cause", "lead to", "result in", "trigger", "provoke"
    2. To predict "no chemical disease induction relation" (Class 0), the text MUST contain explicit non-causal keywords:
       - Non-causal terms: "prevent", "treat", "cure", "associate with", "correlate with", "unrelated to"
    3. If NONE of the above keywords exist in the text → DO NOT modify the original prediction; keep it unchanged.
    """

    prompt = f"""
    Task: Classify the relation between each entity pair in the text as either "chemical induced disease" (Class 1) or "no chemical disease induction relation" (Class 0).
    Original Text: {text}
    Entity Pairs to Check (only low-confidence predictions):
    {chr(10).join(low_confidence_pairs)}
    {evidence_rules}
    ### Output Format:
    Only return the indices of entity pairs that need modification and their new labels, in the format: [(index1, new_label), (index2, new_label)]. 
    - "index" refers to the number of the entity pair (e.g., use 1 for "Entity Pair 1").
    - "new_label" must be 0 (for "no relation") or 1 (for "induced relation").
    Do NOT include any explanations. If no modification is needed, return an empty list: [].
    """
    return prompt.strip()


def apply_confidence_filter(original_pred, corrected_pred, threshold=0.8):
    final_pred = original_pred.copy()
    num_triples = original_pred.shape[0]

    for i in range(num_triples):
        confidence = np.max(original_pred[i])
        if confidence < threshold:
            final_pred[i] = np.zeros(2)
            final_pred[i][corrected_pred[i]] = 1.0
    return final_pred

@hydra.main(config_path="config", config_name="train_docred.yaml", version_base="1.3")
def main(cfg):
    print_config_tree(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    log.info('Creating or Loading DataModule')
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)()

    log.info("Creating DocRE Model")
    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)()


    data_handler = DataHandlerRE(cfg, tokenizer, datamodule)
    log.info(
        f"Data Loaded: Train Samples={len(data_handler.get_train_samples())}, Dev Samples={len(data_handler.get_dev_samples())}")
    react_memory = None
    if cfg.react_memory.enable:
        log.info("尝试加载ReAct记忆...")
        load_success = load_memory_from_cfg(data_handler, cfg)
        react_memory = ReAct_Memory(data_handler)
        react_memory.data_handler1.correct_memory = data_handler.correct_memory
        react_memory.data_handler1.reflexion_memory = data_handler.reflexion_memory
        log.info("ReAct_Memory Initialized (with CorrectMemory/ReflexionMemory)")
    else:
        log.info("ReAct_Memory Disabled")


    if cfg.load_checkpoint:
        log.info("Training from checkpoint")
        model.load_state_dict(torch.load(cfg.load_checkpoint))
        train(cfg,data_handler, datamodule, model, react_memory)
    elif cfg.load_path:
        model.load_state_dict(torch.load(cfg.load_path))
        dev_score, dev_output = evaluate(cfg, model, datamodule.dev_dataset, datamodule.dev_dataloader(),react_memory, epoch=0,tag="dev")
        print(dev_output)
        test_score, test_output = evaluate(cfg, model, datamodule.test_dataset, datamodule.test_dataloader(),react_memory, epoch=0 ,tag="test")
        print(test_output)
    else:
        log.info("Training from scratch")
        train(cfg,data_handler, datamodule, model, react_memory)
    log.info("Finish Training or Testing")


if __name__ == "__main__":
    main()
