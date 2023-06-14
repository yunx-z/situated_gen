import json

from dataset_preprocessor.base_preprocessor import BasePreprocessor
from utils import QA2D


class MCQAPreprocessor(BasePreprocessor):
    def __init__(self):
        super(MCQAPreprocessor, self).__init__()
        self.qa2d = QA2D()

    def read_raw_data(self):
        return super().read_raw_data()

    def pre_pipeline(self, items):
        qa = [(item['_question'], item['_answer']) for item in items]
        statements = self.qa2d.process_multi(qa)
        items = [{"id": item['_id'], "statement": statements[i], "question": item['_question'], "answer": item['_answer']} for i, item in enumerate(items) if statements[i] is not None]
        return items

    def process_json_line(self, json_line, _id):
        item = json.loads(json_line)
        item["_id"] = _id
        item["_question"] = item['question']['stem']
        for choice in item['question']['choices']:
            if choice['label'] == item['answerKey']:
                item["_answer"] = choice['text']
                break
        return item


class CommonsenseQAPreprocessor(MCQAPreprocessor):
    def __init__(self):
        super(CommonsenseQAPreprocessor, self).__init__()
        self.dataset_name = "commonsenseqa"

    def read_raw_data(self):
        items = []
        for split in ["train", "dev"]:
            read_dir = f"data/raw/{self.dataset_name}/{split}_rand_split.jsonl"
            with open(read_dir, 'r') as reader:
                for i, line in enumerate(reader.readlines()):
                    _id = f"{self.dataset_name}::{split}::{i}"
                    item = self.process_json_line(line, _id)
                    items.append(item)
        return items

class ARCPreprocessor(MCQAPreprocessor):
    def __init__(self):
        super(ARCPreprocessor, self).__init__()
        self.dataset_name = "arc"

    def read_raw_data(self):
        items = []
        for _set in ["Challenge", "Easy"]: 
            for split in ["Train", "Dev", "Test"]:
                read_dir = f"data/raw/{self.dataset_name}/ARC-{_set}/ARC-{_set}-{split}.jsonl"
                with open(read_dir, 'r') as reader:
                    for i, line in enumerate(reader.readlines()):
                        _id = f"{self.dataset_name}::{_set}::{split}::{i}"
                        item = self.process_json_line(line, _id)
                        items.append(item)
        return items


