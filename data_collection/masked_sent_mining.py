import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import re
import itertools
import json
import logging
from glob import glob
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from collections import defaultdict
from nltk import sent_tokenize
from tqdm import tqdm

from utils import NER, extarct_nouns, safe_replace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MaskedParaphraseMining:
    def __init__(self, corpus):
        #self.model = SentenceTransformer('model_save/all-mpnet-base-v2')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ner = NER()
        self.logger = logging.getLogger(__name__)
        self.masked_sim_threshold = 0.80
        self.CORPUS = corpus
        assert self.CORPUS in ["QA_DATASETS", "WIKI", "SIMPLEWIKI"]
        self.outfile_name = f"data/preprocessed/postprocessed/pairs_{self.CORPUS}.json"

    """
    def mask_multi(self, sents):
        multi_entities = self.ner.process_multi(sents)
        masked_sents = []
        ori_sents_idx = []
        for i, sent in enumerate(tqdm(sents)):
            entities = multi_entities[i]
            GEO_entities = [ent for ent in entities if ent[1] == 'GPE']
            if len(GEO_entities) == 0:
                continue
            masked_sent = sent
            for ent in GEO_entities:
                masked_sent = masked_sent.replace(ent[0], "[UNK]") # replace whole word
            masked_sent = re.sub(' +', ' ', masked_sent)
            masked_sents.append(masked_sent)
            ori_sents_idx.append(i)
        self.logger.info(f"After GEO-masked: {len(masked_sents)} sentences")
        return masked_sents, ori_sents_idx
    """

    def mask_multi(self, sents_dict):
        sents = [sd['statement'] for sd in sents_dict]
        multi_entities = self.ner.process_multi(sents)
        masked_sents_dict = []
        # ori_sents_idx = []
        for i, sent in enumerate(tqdm(sents)):
            entities = multi_entities[i]
            nouns = extarct_nouns(sent)
            GEO_TEMP_entities = [ent for ent in entities if self.which_type(ent) in ["GEO", "TEMP"]]
            if len(GEO_TEMP_entities) == 0:
                continue
            masked_phrases = []
            masked_sent = sent
            # for ent in GEO_TEMP_entities:
            for ent in entities:
                if ent[0] in masked_sent:
                    masked_phrases.append(ent[0])
                    masked_sent = safe_replace(ent[0], "[UNK]", masked_sent) # replace whole word
            for noun in nouns:
                if noun in masked_sent:
                    masked_phrases.append(noun)
                    masked_sent = safe_replace(noun, "[UNK]", masked_sent)
            # masked_sent = re.sub(' +', ' ', masked_sent)
            masked_sent_dict = sents_dict[i]
            masked_sent_dict['masked_statement'] = masked_sent
            masked_sent_dict['masked_phrases'] = masked_phrases
            masked_sent_dict['GEO_TEMP_entities'] = GEO_TEMP_entities
            masked_sents_dict.append(masked_sent_dict)
            # ori_sents_idx.append(i)
        self.logger.info(f"After N&NE-masked: {len(masked_sents_dict)} sentences")
        # return masked_sents, ori_sents_idx
        return masked_sents_dict


    def extract_sent(self, text):
        sents = sent_tokenize(text)
        filtered_sents = []
        for s in sents:
            if len(s.split()) > 20:
                continue
            if s[-1] not in '.?!”"':
                continue
            if '\n' in s:
                continue
            if not s[0].isupper():
                continue
            filtered_sents.append({'id': self.CORPUS, 'statement': s})
        return filtered_sents

    def load_sentences(self):
        sents = []
        if self.CORPUS == "SIMPLEWIKI" or self.CORPUS == "WIKI":
            dataset_split = "20200501.simple" if self.CORPUS == "SIMPLEWIKI" else "20200501.en"
            dataset = load_dataset("wikipedia", dataset_split)
            for idx in tqdm(range(len(dataset['train']))):
            # for idx in tqdm(range(1000)): # DEBUG
                item = dataset['train'][idx]
                sents.extend(self.extract_sent(item['text']))
        elif self.CORPUS == "QA_DATASETS":
            for file_name in glob("data/preprocessed/statements/*.json"):
                with open(file_name, 'r') as reader:
                    for row in reader:
                        row = json.loads(row)
                        sent = row['statement'].strip()
                        id_ = row['id']
                        sents.append({'id': id_, 'statement': sent})
        self.logger.info(f"Total: {len(sents)} sentences")
        return sents

    def is_in(self, str1, str2):
        str1 = str1.lower()
        str2 = str2.lower()
        if str1 in str2 or str2 in str1:
            return True
        else:
            return False

    def which_type(self, entity):
        tag = entity[1]
        if tag in ['GPE']:
            return 'GEO'
        elif tag in ["DATE", "TIME", "EVENT"]:
            return 'TEMP'
        else:
            return 'OTEHR'

    def is_valid_pair(self, s1, s2):
        s1_statement = s1['statement']
        s2_statement = s2['statement']
        if self.is_in(s1_statement, s2_statement):
            return None
        # not too much masked token
        s1_masked_cnt = s1['masked_statement'].count('[UNK]')
        s2_masked_cnt = s2['masked_statement'].count('[UNK]')
        if s1_masked_cnt > 5 or s2_masked_cnt > 5:
            return None
        if s1_masked_cnt < 2 or s2_masked_cnt < 2:
            return None

        s1_entities = s1['GEO_TEMP_entities']
        s2_entities = s2['GEO_TEMP_entities']
        # no overlapping entities
        """
        for s1_entity, s2_entity in itertools.product(s1_entities, s2_entities):
            if self.is_in(s1_entity[0], s2_entity[0]):
                return None
        """
        for s1_entity in s1_entities:
            if self.is_in(s1_entity[0], s2_statement):
                return None
        for s2_entity in s2_entities:
            if self.is_in(s2_entity[0], s1_statement):
                return None

        # same type
        s1_types = set([self.which_type(s1_entity) for s1_entity in s1_entities]) 
        s2_types = set([self.which_type(s2_entity) for s2_entity in s2_entities])
        intersect_s1_s2_types = s1_types & s2_types
        if "GEO" in intersect_s1_s2_types and "TEMP" in intersect_s1_s2_types:
            return ["GEO", "TEMP"]
        elif "GEO" in intersect_s1_s2_types:
            return ["GEO"]
        elif "TEMP" in intersect_s1_s2_types:
            return ["TEMP"]
        else:
            return None

    def run(self):
        self.logger.info(f"load and preprocess {self.CORPUS} sentences...")
        sents = self.load_sentences()
        self.logger.info("apply NER and mask sentences...")
        # masked_sents, ori_sents_idx = self.mask_multi(sents)
        masked_sents = self.mask_multi(sents)
        self.logger.info("buil inverse dict...")
        inverse_dict = defaultdict(list)
        # for masked_sent, ori_sent_idx in tqdm(zip(masked_sents, ori_sents_idx), total=len(masked_sents)):
        for masked_sent in masked_sents:
            # inverse_dict[masked_sent].append(sents[ori_sent_idx])
            inverse_dict[masked_sent['masked_statement']].append(masked_sent)
        self.logger.info("masked paraphrase mining...")
        masked_sents_str = list(inverse_dict.keys())
        masked_paraphrases = util.paraphrase_mining(self.model, masked_sents_str, show_progress_bar=True, batch_size=256, max_pairs=100000)
        self.logger.info(f"saving to {self.outfile_name}")
        with open(self.outfile_name, 'w') as writer:
            for masked_statement, statements in inverse_dict.items():
                if len(statements) > 1:
                    for pair in itertools.combinations(statements, 2):
                        pair_type = self.is_valid_pair(*pair)
                        if pair_type is not None:
                            item = {'masked_sim_score' : 1,
                                    'pair_type' : pair_type,
                                    'statement_1' : pair[0],
                                    'statement_2' : pair[1]
                                    }
                            writer.write(json.dumps(item)+'\n')
            for paraphrase in tqdm(masked_paraphrases):
                score, i, j = paraphrase
                if score < self.masked_sim_threshold:
                    continue
                statement_1_masked = masked_sents_str[i]
                statement_2_masked = masked_sents_str[j]
                statements_1 = inverse_dict[statement_1_masked]
                statements_2 = inverse_dict[statement_2_masked]
                for statement_1, statement_2 in itertools.product(statements_1, statements_2):
                    pair_type = self.is_valid_pair(statement_1, statement_2)
                    if pair_type is not None:
                        item = dict()
                        item['masked_sim_score'] = score
                        item['pair_type'] = pair_type
                        item['statement_1'] = statement_1
                        item['statement_2'] = statement_2
                        writer.write(json.dumps(item)+'\n')




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="corpus name")
    args = parser.parse_args()
    corpus = args.corpus
    worker = MaskedParaphraseMining(corpus)
    worker.run()
