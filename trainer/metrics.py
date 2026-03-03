from logging import getLogger
logger = getLogger('train_logger')

from data_utils.data_handler_re import DataHandlerRE


SUCCESS = 0
FILE_ERROR = 1
NOT_ZIP_FILE = 2
ENCODING_ERROR = 3
JSON_ERROR = 4
SCHEMA_ERROR = 5
ALIAS_FORMAT_ERROR = 6


class Metric:
    def __init__(self, data_handler: DataHandlerRE):
        self.data_handler = data_handler

    def evaluate(self):
        predict_result = self.load_predict_result()
        golden_dict = self.load_golden_dict()
        alias_dict = {}
        ret_info = self.eval(predict_result, golden_dict, alias_dict)
        return ret_info

    def eval(self, predict_result, golden_dict, alias_dict={}):
        ret_info = {}
        correct_sum, predict_sum, recall_sum, recall_correct_sum = 0.0, 0.0, 0.0, 0.0
        for sent in golden_dict:
            golden_spo_list = self.del_duplicate(golden_dict[sent], alias_dict)
            predict_spo_list = predict_result.get(sent, list())
            normalized_predict_spo = self.del_duplicate(predict_spo_list, alias_dict)
            recall_sum += len(golden_spo_list)
            predict_sum += len(normalized_predict_spo)
            for spo in normalized_predict_spo:
                if self.is_spo_in_list(spo, golden_spo_list, alias_dict):
                    correct_sum += 1
            for golden_spo in golden_spo_list:
                if self.is_spo_in_list(golden_spo, predict_spo_list, alias_dict):
                    recall_correct_sum += 1
        logger.info("correct spo num = {}".format(correct_sum))
        logger.info("submitted spo num = {}".format(predict_sum))
        logger.info("golden set spo num = {}".format(recall_sum))
        logger.info("submitted recall spo num = {}".format(recall_correct_sum))
        precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
        recall = recall_correct_sum / recall_sum if recall_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        ret_info["data"] = []
        ret_info["data"].append({"name": "precision", "value": precision})
        ret_info["data"].append({"name": "recall", "value": recall})
        ret_info["data"].append({"name": "f1-score", "value": f1})
        return ret_info

    def load_predict_result(self):
        predict_result = {}
        for line in self.data_handler.ds_pred:
            sent = line["input"]
            spo_list = line["spo_list_pred"]
            predict_result[sent] = spo_list
        return predict_result

    def load_golden_dict(self):
        golden_dict = {}
        for line in self.data_handler.ds_test:
            sent = line["input"]
            spo_list = eval(line["output"])
            golden_dict[sent] = spo_list
        return golden_dict

    @staticmethod
    def del_bookname(entity_name):
        if entity_name.startswith("《") and entity_name.endswith("》"):
            entity_name = entity_name[1:-1]
        return entity_name

    @staticmethod
    def del_duplicate(spo_list, alias_dict):
        normalized_spo_list = []
        for spo in spo_list:
            if not Metric.is_spo_in_list(spo, normalized_spo_list, alias_dict):
                normalized_spo_list.append(spo)
        return normalized_spo_list

    @staticmethod
    def is_spo_in_list(target_spo, golden_spo_list, alias_dict):
        if target_spo in golden_spo_list:
            return True
        target_s = target_spo["subject"]
        target_p = target_spo["predicate"]
        target_o = target_spo["object"]
        target_s_alias_set = alias_dict.get(target_s, set())
        target_s_alias_set.add(target_s)
        target_o_alias_set = alias_dict.get(target_o, set())
        target_o_alias_set.add(target_o)
        for spo in golden_spo_list:
            s = spo["subject"]
            p = spo["predicate"]
            o = spo["object"]
            if p != target_p:
                continue
            # if s in target_s_alias_set and _is_equal_o(o, target_o, alias_dict):
            if s in target_s_alias_set and o in target_o_alias_set:
                return True
        return False

if __name__ == '__main__':
    metric = Metric()