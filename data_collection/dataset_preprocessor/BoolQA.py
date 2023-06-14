import json

from dataset_preprocessor.base_preprocessor import BasePreprocessor
from utils import BoolQ2D


class BoolQAPreprocessor(BasePreprocessor):
    def __init__(self):
        super(BoolQAPreprocessor, self).__init__()
        self.boolq2d = BoolQ2D()

    def read_raw_data(self):
        return super().read_raw_data()

    def pre_pipeline(self, items):
        boolqa = [(item['_question'], item['_answer']) for item in items]
        statements = self.boolq2d.process_multi(boolqa)
        items = [{"id": item['_id'], "statement": statements[i], "question": item['_question'], "answer": item['_answer']} for i, item in enumerate(items) if statements[i] is not None]
        return items


class StrategyQAPreprocessor(BoolQAPreprocessor):
    def __init__(self):
        super(StrategyQAPreprocessor, self).__init__()
        self.dataset_name = "strategyqa"

    def read_raw_data(self):
        items = []
        for split in ["train", "train_filtered"]:
            read_dir = f"data/raw/{self.dataset_name}/strategyqa_{split}.json"
            with open(read_dir, 'r') as reader:
                json_list = json.loads(reader.read())
            for i, item in enumerate(json_list):
                item["_id"] = f"{self.dataset_name}::{split}::{i}"
                item["_question"] = item['question']
                item["_answer"] = item['answer']
                items.append(item)
        return items


