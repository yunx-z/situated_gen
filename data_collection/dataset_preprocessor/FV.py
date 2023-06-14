import json
import csv
from tqdm import tqdm

from dataset_preprocessor.base_preprocessor import BasePreprocessor


class FVPreprocessor(BasePreprocessor):
    def __init__(self):
        super(FVPreprocessor, self).__init__()

    def read_raw_data(self):
        return super().read_raw_data()


class CREAKPreprocessor(FVPreprocessor):
    def __init__(self):
        super(CREAKPreprocessor, self).__init__()
        self.dataset_name = "creak"

    def read_raw_data(self):
        items = []
        for split in ["train", "dev"]:
            read_dir = f"data/raw/{self.dataset_name}/{split}.json"
            with open(read_dir, 'r') as reader:
                for i, line in enumerate(reader.readlines()):
                    item = json.loads(line)
                    item["_id"] = f"{self.dataset_name}::{split}::{i}"
                    items.append(item)
        return items

    def pre_pipeline(self, items):
        # remove false claims
        items = [{"id": item['_id'], "statement": item['sentence']} for item in items if item["label"] == "true"]
        return items


class OpenbookQAPreprocessor(FVPreprocessor):
    def __init__(self):
        super(OpenbookQAPreprocessor, self).__init__()
        self.dataset_name = "openbookqa"

    def read_raw_data(self):
        items = []
        for file_name in ["Additional/crowdsourced-facts.txt", "Main/openbook.txt"]:
            read_dir = f"data/raw/{self.dataset_name}/Data/{file_name}"
            with open(read_dir, 'r') as reader:
                for i, line in enumerate(reader.readlines()):
                    item = dict()
                    item["id"] = f"{self.dataset_name}::{file_name}::{i}"
                    item["statement"] = self.clean(line)
                    items.append(item)
        return items

    def clean(self, line):
        if line[0] == '"' and line[-1] == '"':
            line = line[1:-1]
        return line


class OMCSPreprocessor(FVPreprocessor):
    def __init__(self):
        super(OMCSPreprocessor, self).__init__()
        self.dataset_name = "omcs"

    def read_raw_data(self):
        items = []
        for file_name in ["omcs-sentences-free.txt", "omcs-sentences-more.txt"]:
            read_dir = f"data/raw/{self.dataset_name}/{file_name}"
            with open(read_dir, newline='') as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t')
                for i, item in enumerate(tqdm(reader)):
                    if item["language_id"] == "en":
                        item["_id"] = f"{self.dataset_name}::{file_name}::{i}"
                        items.append(item)
        return items

    def pre_pipeline(self, items):
        items = [{"id": item['_id'], "statement": item['text']} for item in items if len(item['text']) < 512]
        return items


class QASCPreprocessor(FVPreprocessor):
    def __init__(self):
        super(QASCPreprocessor, self).__init__()
        self.dataset_name = "qasc"

    def read_raw_data(self):
        items = []
        read_dir = f"data/raw/{self.dataset_name}/QASC_Corpus.txt"
        with open(read_dir, 'r') as reader:
            for i, line in enumerate(tqdm(reader)):
                item = dict()
                item["_id"] = f"{self.dataset_name}::corpus::{i}"
                item["text"] = line
                items.append(item)
        return items

    def pre_pipeline(self, items):
        items = [{"id": item['_id'], "statement": item['text']} for item in items if len(item['text']) < 512]
        return items


