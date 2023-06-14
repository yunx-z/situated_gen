import os
import itertools
import random
import json
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from communities.algorithms import louvain_method
from communities.visualization import draw_communities

from utils import safe_replace, shuffle_two_lists_together

random.seed(17)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DatasetBuilder:
    def __init__(self, corpus):
        self.logger = logging.getLogger(__name__)
        self.CORPUS = corpus
        assert self.CORPUS in ["QA_DATASETS", "WIKI", "SIMPLEWIKI"]
        self.infile_name = f"data/preprocessed/postprocessed/pairs_{self.CORPUS}.json"
        self.ratio = {'test': 0.25, 'dev': 0.15}
        self.outdir = f"data/final"

    def run(self):
        raw_items = self.load_data(self.infile_name)
        self.logger.info("split dataset ...")
        # raw_datasets = self.split_train_dev_test(raw_items)
        raw_datasets = self.split_train_dev_test(raw_items)
        task2builder = {
                'boolqa' : self.builder_boolqa,
                'keyword2text' : self.builder_keyword2text,
                }
        datasets = dict()
        self.logger.info("build dataset ...")
        for task, builder in task2builder.items():
            datasets[task] = dict()
            for split, items in raw_datasets.items():
                self.logger.info(f"Task: {task}; Split: {split}")
                datasets[task][split] = list()
                for item in tqdm(items):
                    datasets[task][split].extend(builder(item))
        self.logger.info("saving dataset ...")
        self.save_datasets(datasets)

    def load_data(self, file_name):
        items = []
        with open(file_name, 'r') as reader:
            for line in reader:
                item = json.loads(line)
                # item = self.preprocess_item(item)
                items.append(item)
        # items = items[:1000] # debug
        self.logger.info(f"load {len(items)} raw items")
        return items

    def preprocess_item(self, item):
        item['statement_1']['statement'] = self.clean_sent(item['statement_1']['statement'])
        item['statement_2']['statement'] = self.clean_sent(item['statement_2']['statement'])
        return item

    def clean_sent(self, sent):
        if sent[-1] not in '.!?"':
            sent += '.'
        if sent[0] == '"' and sent[-1] == '"':
            sent = sent[1:-1]
        sent = sent[0].upper() + sent[1:]
        return sent

    def rand_split_train_dev_test(self, items):
        random.shuffle(items)
        dev_cnt = len(items) * self.ratio['dev']
        test_cnt = len(items) * self.ratio['test']
        datasets = {'train':items[dev_cnt+test_cnt:], 'dev':items[:dev_cnt], 'test':items[dev_cnt:dev_cnt+test_cnt]}
        return datasets

    def split_train_dev_test(self, items):
        idpair2item = dict()
        G = nx.Graph()
        datasets = {'train': [], 'dev': [], 'test': []}
        for item in items:
            id1 = item['statement_1']['id']
            id2 = item['statement_2']['id']
            idpair2item[(id1, id2)] = item
            idpair2item[(id2, id1)] = item
            G.add_edge(id1, id2)
        ids = list(G.nodes)
        adj_matrix = nx.to_numpy_array(G, nodelist=ids, dtype=np.int64)
        communities, _ = louvain_method(adj_matrix)
        # draw_communities(adj_matrix, communities)
        # plt.savefig("tmp/communities.png")
        S = [G.subgraph([ids[index] for index in c]).copy() for c in communities]
        # S = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=False)]
        # 2022-05-17 12:05:15,176 - __main__ - INFO - pair stats (subgraph_edge_cnt:occurence): [(1, 31), (2, 12), (3, 1), (4, 2), (9, 1), (12, 1), (15, 1), (23, 1), (142, 1), (12443, 1)]
        # 2022-05-17 12:05:15,176 - __main__ - INFO - unique statement stats (subgraph_node_cnt:occurence): [(2, 31), (3, 13), (4, 1), (5, 1), (6, 1), (7, 1), (9, 1), (11, 1), (31, 1), (1444, 1)]
        S_edge_cnt = [len(s.edges) for s in S]
        S_node_cnt = [len(s.nodes) for s in S]
        self.logger.info(f"pair stats (subgraph_edge_cnt:occurence): {sorted(Counter(S_edge_cnt).items())}")
        self.logger.info(f"unique statement stats (subgraph_node_cnt:occurence): {sorted(Counter(S_node_cnt).items())}")
        
        total_cnt = sum(S_edge_cnt)
        self.logger.info(f"{total_cnt} items left after community separating")
        random.shuffle(S)
        for subgraph in S:
            add_to_train = True
            subitems = [idpair2item[e] for e in subgraph.edges]
            for split in self.ratio:
                if len(datasets[split]) + len(subitems) <= total_cnt * self.ratio[split]:
                    add_to_train = False
                    datasets[split].extend(subitems)
                    break
            if add_to_train:
                datasets['train'].extend(subitems)
        # datasets = {'train':items[:10], 'dev':items[10:20], 'test':items[20:]}
        assert total_cnt == sum([len(items) for split, items in datasets.items()])
        for split, items in datasets.items():
            self.logger.info(f"Raw dataset '{split}' split has {len(items)} items/pairs")
        assert self.check_datasets(datasets)
        return datasets

    def check_datasets(self, datasets):
        split_ids = {'train': set(), 'dev': set(), 'test': set()}
        for split, items in datasets.items():
            for item in items:
                split_ids[split].add(item['statement_1']['id'])
                split_ids[split].add(item['statement_2']['id'])
        for split, ids in split_ids.items():
            self.logger.info(f"Raw dataset '{split}' split has {len(ids)} unqiue statements")
        # no intersects between train/dev/test statements(nodes)
        for split1, split2 in itertools.combinations(list(split_ids.keys()), 2):
            if len(split_ids[split1] & split_ids[split2]) != 0:
                return False
        return True

    def which_type(self, entity):
        tag = entity[1]
        if tag in ['GPE']:
            return 'GEO'
        elif tag in ["DATE", "TIME", "EVENT"]:
            return 'TEMP'
        else:
            return 'OTEHR'

    def builder_boolqa(self, item):
        dataset_items = []
        s1 = item['statement_1']
        s2 = item['statement_2']
        dataset_items.append({'statement': s1['statement'], 'label': "True"})
        dataset_items.append({'statement': s2['statement'], 'label': "True"})
        # swap GEO/TEMP entities to generate false statements
        for pair_type in item['pair_type']:
            s1_entity = random.choice([e for e in s1['GEO_TEMP_entities'] if self.which_type(e) == pair_type])
            s2_entity = random.choice([e for e in s2['GEO_TEMP_entities'] if self.which_type(e) == pair_type])
            try:
                assert s1_entity[0] in s1['statement'] 
                assert s2_entity[0] in s2['statement'] 
                assert s1_entity[0] not in s2['statement']
                assert s2_entity[0] not in s1['statement']
            except Exception as e:
                self.logger.warning(f"Entity inclusion error:\n{s1_entity[0]} - {s1['statement']}\n{s2_entity[0]} - {s2['statement']}")
            else:
                s1_false_statement = safe_replace(s1_entity[0], s2_entity[0], s1['statement'])
                s2_false_statement = safe_replace(s2_entity[0], s1_entity[0], s2['statement'])
                assert s1['statement'] != s1_false_statement
                assert s2['statement'] != s2_false_statement
                dataset_items.append({'statement': s1_false_statement, 'label': "False"})
                dataset_items.append({'statement': s2_false_statement, 'label': "False"})
        return dataset_items
            
    def builder_keyword2text(self, item):
        dataset_items = []
        s1 = item['statement_1']
        s2 = item['statement_2']
        s1_keywords = s1['masked_phrases']
        s2_keywords = s2['masked_phrases']
        keywords = s1_keywords + s2_keywords
        keywords_pos = [0 for i in range(len(s1_keywords))] + [1 for i in range(len(s2_keywords))]
        # random.shuffle(keywords)
        keywords, keywords_pos = shuffle_two_lists_together(keywords, keywords_pos) 
        s1_statement = self.clean_sent(s1['statement'])
        s2_statement = self.clean_sent(s2['statement'])
        dataset_items.append({'keywords': keywords, 'statement': s1_statement + ' ' + s2_statement, 'ids': [s1['id'], s2['id']], 'keywords_pos': keywords_pos, 'statements': [s1_statement, s2_statement]})
        dataset_items.append({'keywords': keywords, 'statement': s2_statement + ' ' + s1_statement, 'ids': [s2['id'], s1['id']], 'keywords_pos': [1-p for p in keywords_pos], 'statements': [s2_statement, s1_statement]})
        return dataset_items

    def save_datasets(self, datasets):
        for task in datasets:
            task_dir = os.path.join(self.outdir, task)
            os.makedirs(task_dir, exist_ok=True)
            for split in datasets[task]:
                split_dir = os.path.join(task_dir, f"{split}.json")
                self.logger.info(f"{len(datasets[task][split])} examples in {split_dir}")
                df = pd.DataFrame(datasets[task][split])
                df['statement'] = df['statement'].map(self.clean_sent)
                df = df.drop_duplicates(subset=['statement'])
                df.to_json(split_dir, orient='records', lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="corpus name")
    args = parser.parse_args()
    corpus = args.corpus
    worker = DatasetBuilder(corpus)
    worker.run()
