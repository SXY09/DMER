import re, json
# from logging import getLogger
# logger = getLogger('train_logger')

from .eval_audit import FormatorUtils
from .eval_audit import AuditVoid, AuditBothEmpty, AuditLabelEmptyOnly, AuditPredEmptyOnly, AuditLong, AuditInsane, AuditRepeat, AuditRetard, AuditWhatever
from .eval_metric import MetricBase, MetricF1


class EvaluatorBase(FormatorUtils):
    def __init__(self):
        self.last = dict()
        self._init_audit()
        self._init_metric()
    
    def _init_metric(self):
        self.metric = MetricBase()

    def _extract(self, golden_list, predict_str: str):
        raise NotImplementedError()

    def _init_audit(self):
        self.audit = [
            AuditVoid(),
            AuditBothEmpty(),
            AuditLabelEmptyOnly(),
            AuditPredEmptyOnly(),
            AuditLong(),
            AuditInsane(),
            AuditRepeat(),
            AuditRetard(),
            AuditWhatever()
        ]
    
    def _update_audit(self):
        for audit in self.audit:
            audit.update(self.last)

    def add(self, golden_list, predict_str):
        if isinstance(golden_list, str):
            golden_list = json.loads(golden_list)
        if not isinstance(predict_str, str):
            predict_str = json.dumps(predict_str)
        y_truth, y_pred = self._extract(golden_list, predict_str)
        self.metric.update(y_truth, y_pred)

        self.last['json_data'] = golden_list
        self.last['predict'] = predict_str
        self.last['y_truth'] = y_truth
        self.last['y_pred'] = y_pred
        self.last['metric'] = self.metric

        self._update_audit()
    
    def add_batch(self, json_data, predict):
        for i, j in zip(json_data, predict):
            self.add(i, j)

    def get_metric(self) -> float:
        return self.metric.get_metric()

    def get_last_metric(self):
        return self.metric.get_last()

    def get_audit_report(self):
        return {
            a.get_name() : a.get_report()
            for a in self.audit
        }
    def dump_audit_report(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_report(), f, indent=4, ensure_ascii=False)


class EvaluatorRE(EvaluatorBase):
    keys = ['subject', 'predicate', 'object']
    def _init_metric(self):
        self.metric = MetricF1()

    def _format_triplet(self, triplet):
        triplet_ = [triplet[k] for k in self.keys]
        triplet_str = "|".join(triplet_)
        return triplet_str

    def _extract(self, golden_list, predict_str):
        y_truth = set()
        for triplet in golden_list:
            triplet_str = self._format_triplet(triplet)
            y_truth.add(self._format(triplet_str))
        y_pred = set()
        predict_str = json.loads(predict_str)
        for triplet in predict_str:
            triplet_str = self._format_triplet(triplet)
            y_pred.add(self._format(triplet_str))
        return y_truth, y_pred

    def get_metric_dict(self):
        f1, recall, precision = self.metric.get_detail()
        f1 = round(f1, 4)
        recall = round(recall, 4)
        precision = round(precision, 4)
        return {
            "f1": f1,
            "recall": recall,
            "precision": precision
        }