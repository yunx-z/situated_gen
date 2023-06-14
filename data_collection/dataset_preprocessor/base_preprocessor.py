import json
from abc import ABC, abstractmethod
from tqdm import tqdm

from utils import NER


class BasePreprocessor(ABC):
    def __init__(self):
        self.dataset_name = None
        self.ner = NER()
        self.GEO_sent_cnt = 0
        self.TEMP_sent_cnt = 0
        self.GEO_TEMP_sent_cnt = 0

    @abstractmethod
    def read_raw_data(self):
        pass

    def pre_pipeline(self, items):
        return items

    def pipeline_statements(self, items):
        sents = [item['statement'] for item in items]
        multi_entities = self.ner.process_multi(sents)
        return_items = []
        for i, item in enumerate(items):
            entities = multi_entities[i]
            GEO_entities = [ent for ent in entities if ent[1] == 'GPE'] 
            TEMP_entities = [ent for ent in entities if ent[1] in ["DATE", "TIME", "EVENT"]] 
            if len(GEO_entities) + len(TEMP_entities) == 0:
                continue
            if len(GEO_entities) > 0:
                self.GEO_sent_cnt += 1
            elif len(TEMP_entities) > 0:
                self.TEMP_sent_cnt += 1
            else:
                raise ValueError("No GEO or TEMP entities!")
            if len(GEO_entities) > 0 and len(TEMP_entities) > 0:
                self.GEO_TEMP_sent_cnt += 1
            entities_text = [ent[0] for ent in entities]
            # temporarily using NER for keyword extraction
            # item['concept_set'] = '#'.join(entities_text)
            # item['scene'] = [sent]
            item['NERs'] = ', '.join([f"{ent[0]}:{ent[1]}" for ent in entities])
            return_items.append(item)
        return return_items
 
    def save(self, items):
        save_dir = f"data/preprocessed/statements/{self.dataset_name}.json"
        with open(save_dir, 'w') as writer:
            for item in items:
                writer.write(json.dumps(item)+'\n')
        print(f"saved to {save_dir}") 

    def run(self):
        items = self.read_raw_data()
        print(f"{self.dataset_name} has {len(items)} examples")
        items = self.pre_pipeline(items)
        print(f"After pre-pipeline filtering, it has {len(items)} examples")
        items = self.pipeline_statements(items)
        print(f"After pipeline filtering and extraction, it has {len(items)} examples")
        self.save(items)
        print("\nStatistics:")
        print("GEO_sent_cnt", self.GEO_sent_cnt)
        print("TEMP_sent_cnt", self.TEMP_sent_cnt)
        print("GEO_TEMP_sent_cnt", self.GEO_TEMP_sent_cnt)

