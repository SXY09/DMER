""" from [InstructUIE] """

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class FormatorUtils:
    @staticmethod
    def _resolve_option(s):
        "s: instruction"
        option_parts = re.findall('Option:(.+?)\n', s)
        if len(option_parts) <= 0:
            return []
        option_part = option_parts[0]
        ans = [FormatorUtils._format(x) for x in option_part.split(',')]
        return ans

    @staticmethod
    def _remove_redundant_space(s):
        s = ' '.join(s.split())
        s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)
        return s
    
    @staticmethod
    def _format(s):
        s = FormatorUtils._remove_redundant_space(s)
        s = s.lower()
        s = s.replace('{','').replace('}','')
        s = re.sub(',+', ',', s)
        s = re.sub('\.+', '.', s)
        s = re.sub(';+', ';', s)
        s = s.replace('’', "'")
        s = s.replace('location', 'located')
        return s

    @staticmethod
    def _format_json_dict(s):
        pattern = re.compile(r'\{.*?\}')
        match = pattern.findall(s)
        res = []
        for m in match:
            res.append(FormatorUtils._format(m))
        return res

    @staticmethod
    def _format_tuple_dict(s):
        pattern = re.compile(r'\(.*?\)')
        match = pattern.findall(s)
        res = []
        for m in match:
            res.append(FormatorUtils._format(m))
        return res

    @staticmethod
    def _re_item(s):
        return re.findall(r'(?:^|\()([^\(\)]+?)(?:$|\))', s.strip())
    
    @staticmethod
    def _resolve_brackets(s):
        ans = []
        level = 0
        last_lb_idx = None
        for idx, char in enumerate(s):
            if char == '(':
                if level == 0:
                    last_lb_idx = idx
                level += 1
            elif char == ')':
                if last_lb_idx is None and len(ans) == 0 and 0 != idx:
                    ans.append(s[0 : idx])
                if level == 1 and last_lb_idx+1 != idx:
                    ans.append(s[last_lb_idx+1 : idx])
                if level >= 1:
                    level -= 1
        if level == 1 and last_lb_idx+1 != len(s):
            ans.append(s[last_lb_idx+1:])
        return ans
    
    @staticmethod
    def _resolve_comma(s):
        ans = []
        level = 0
        last_comma = -1
        for idx, char in enumerate(s):
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            elif char == ',' and level == 0 and last_comma + 1 != idx:
                ans.append(s[last_comma+1 : idx])
                last_comma = idx
        if last_comma+1 != len(s):
            ans.append(s[last_comma+1:])
        return ans


class AuditBase:
    def __init__(self, record_limit=16):
        self.record_limit = record_limit
        self.cnt = 0
        self.record = []
    def _check(self, last) -> bool:
        raise NotImplementedError()
    def _add_record(self, new_record):
        self.cnt += 1
        if self.record_limit < 0 or len(self.record) < self.record_limit:
            self.record.append(new_record)
        elif os.environ.get('RANDOM_RECORD')=='1':
            if random.randint(1,self.cnt) <= self.record_limit:
                idx = random.randint(0,len(self.record)-1)
                self.record[idx] = new_record
    def update(self, last):
        if self._check(last):
            new_record = {
                'json_data': last['json_data'],
                'predict': last['predict'],
                'y_truth': last['y_truth'],
                'y_pred': last['y_pred']
            }
            new_record = self._to_json_object(new_record)
            self._add_record(new_record)
    @staticmethod
    def _to_json_object(obj):
        if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float):
            return obj
        if isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, set):
            return [AuditBase._to_json_object(x) for x in obj]
        if isinstance(obj, dict):
            return {AuditBase._to_json_object(k): AuditBase._to_json_object(v) for k, v in obj.items()}
        else:
            raise NotImplementedError()
    def get_cnt(self):
        return self.cnt
    def get_record(self):
        return self.record
    def get_report(self):
        return {
            'count': self.cnt,
            'record': self.record
        }
    def get_name(self):
        return self.__class__.__name__

class AuditVoid(AuditBase):
    def _check(self, last) -> bool:
        return last['predict'].strip() == ''

class AuditLong(AuditBase):
    def _check(self, last) -> bool:
        return len(last['predict']) >= 512     # 长度上限根据需要自行修改

class AuditInsane(AuditBase):
    def _check(self, last) -> bool:
        return last['predict'].strip().lower() not in {'na', 'no relation', 'none', '[]', ''} and len(last['y_pred']) == 0    # 说了点什么，但又什么有用的都没说

class AuditBothEmpty(AuditBase):
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) == 0

class AuditLabelEmptyOnly(AuditBase):
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) != 0

class AuditPredEmptyOnly(AuditBase):
    def _check(self, last) -> bool:
        return len(last['y_truth']) != 0 and len(last['y_pred']) == 0
    
class AuditNA(AuditBase):
    def _check(self, last) -> bool:
        for i in last['y_pred']:    # assert isinstance(i, str)
            if ',na,' in i:
                return True
        return False

class AuditInvalid(AuditBase):
    def _check(self, last) -> bool:
        valid_labels = FormatorUtils._resolve_option(last['json_data']['Instance']['instruction'])
        if len(valid_labels) == 0:
            return False
        valid_labels = set(valid_labels)

        for pred in last['y_pred']:
            pred = pred.split(':')
            if len(pred) >= 2:
                label = pred[0]
                if label not in valid_labels:
                    return True
        return False

class AuditFidelity(AuditBase):
    def _check(self, last) -> bool:
        for item in last['y_pred']:
            item = item.split(':')       #   这里对于实体或标签本身就包含逗号的情况不好处理，
            if len(item) < 2:
                continue
            ents = item[-1].split(',')
            for ent in ents:
                if FormatorUtils._format(ent) not in FormatorUtils._format(last['json_data']['Instance']['sentence']):
                    return True
            return False


class AuditGoldenlabelFault(AuditBase):
    def _check(self, last) -> bool:
        for item in last['y_truth']:
            cnt = 0
            if len(item.split(':')) < 2:
                continue
            for i in item.split(':')[-1].split(','):
                i = i.strip()
                if i != '':
                    cnt += 1
            if cnt <= 1:
                return True
        return False

class AuditRepeat(AuditBase):
    def _check(self, last) -> bool:
        pattern = r'(\w{5,})\1{2,}'
        match = re.search(pattern, last['predict'])
        return match is not None

class AuditRetard(AuditBase):
    def _check(self, last) -> bool:
        last_metric = last['metric']
        if hasattr(last_metric, 'last_TP'):
            if len(last['y_pred']) != 0 and len(last['y_truth']) != 0:
                return last_metric.last_TP == 0
        if hasattr(last_metric, 'scores'):
            return last_metric.scores[-1] == 0
        return False
        
class AuditWhatever(AuditBase):
    def _check(self, last) -> bool:
        return True
    
class AuditConfuseMatrix(AuditBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = None
        self.options2idx = None
        self.matrix = None
        self.dataset_name = None
    def _check(self, last) -> bool:
        raise NotImplementedError()
    @staticmethod
    def _resolve(s):
        s = [i.strip() for i in s.split(',')]
        if len(s) == 2:
            return s[0], s[1]
        elif len(s) == 3:
            return '%s,%s'%(s[0],s[2]), s[1]
        else:
            return None
    def update(self, last):
        if self.dataset_name is None:
            self.dataset_name = last['json_data']['Dataset']
            self.options = FormatorUtils._resolve_option(last['json_data']['Instance']['instruction'])
            if 'na' not in self.options:
                self.options.append('na')
            self.options2idx = dict()
            for idx, option in enumerate(self.options):
                self.options2idx[option] = idx
            N = len(self.options)
            self.matrix = np.zeros((N, N), dtype=np.int16)
        truth = dict()
        pred = dict()
        is_conflict = False
        for item in last['y_truth']:
            res = self._resolve(item)
            if res is not None:
                if res[0] in truth:
                    is_conflict = True
                truth[res[0]] = res[1]
        for item in last['y_pred']:
            res = self._resolve(item)
            if res is not None:
                if res[0] in pred:
                    is_conflict = True
                pred[res[0]] = res[1]
        for k in truth:
            if truth[k] not in self.options2idx:
                continue
            idx_truth = self.options2idx[truth[k]]
            if k in pred:
                if pred[k] in self.options2idx:
                    idx_pred = self.options2idx[pred[k]]
                    self.matrix[idx_truth][idx_pred] += 1
            else:
                idx_pred = self.options2idx['na']
                self.matrix[idx_truth][idx_pred] += 1
        for k in pred:
            if pred[k] not in self.options2idx:
                continue
            idx_pred = self.options2idx[pred[k]]
            if k not in truth:
                idx_truth = self.options2idx['na']
                self.matrix[idx_truth][idx_pred] += 1

        if is_conflict:
            new_record = {
                'json_data': last['json_data'],
                'predict': last['predict'],
                'y_truth': list(last['y_truth']),
                'y_pred': list(last['y_pred'])
            }
            new_record = self._to_json_object(new_record)
            self._add_record(new_record)

    def get_report(self):
        if os.environ.get('EXPORT_IMG') == '1':
            root = 'img'
            if not os.path.exists(root):
                os.mkdir(root)
            fpath = os.path.join(root, '%s.png'%self.dataset_name)
            if True:
                matrix = ((1-np.eye(self.matrix.shape[0])) * self.matrix).astype(np.int16)    
                na = self.options2idx['na']
                matrix[na,:]=0
                matrix[:,na]=0
            else:
                matrix = self.matrix
            self._plot_matrix(matrix, self.options, fpath, title=self.dataset_name)
        return super().get_report()
    @staticmethod
    def _plot_matrix(A, labels, fpath, title=None, min_size = 50):
        N = len(labels)
        figsize = N * min_size / 100 + 1, N * min_size / 100 + 1
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(A, cmap='viridis')

        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(N):
            for j in range(N):
                text = ax.text(j, i, A[i, j], ha='center', va='center', color='w')
        
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fpath)
        plt.close()
    