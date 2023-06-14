import re
import string
import spacy
import random
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from pycorenlp import StanfordCoreNLP

random.seed(17)
nlp = spacy.load('en_core_web_sm')

class NER:
    def __init__(self):
        self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    def process_one(self, sent):
        entities = []
        sentence = Sentence(sent)
        self.tagger.predict(sentence)
        ents = sentence.to_dict(tag_type='ner')
        for ent in ents['entities']:
            ent_label = ent['labels'][0].value
            entities.append((ent['text'], ent_label))
        return entities
   
    def process_multi(self, sents, batch_sz=64):
        multi_entities = []
        for sent in tqdm(sents):
            doc = nlp(sent)
            entities = []
            if doc.ents:
                for ent in doc.ents:
                    entities.append((ent.text, ent.label_))
            multi_entities.append(entities)
        return multi_entities

    """
    def process_multi(self, sents, batch_sz=64):
        sentences = [Sentence(sent) for sent in sents]
        self.tagger.predict(sentences, mini_batch_size=batch_sz, verbose=True)
        multi_entities = []
        for sentence in sentences:
            entities = []
            ents = sentence.to_dict(tag_type='ner')
            for ent in ents['entities']:
                ent_label = ent['labels'][0].value
                entities.append((ent['text'], ent_label))
            multi_entities.append(entities)
        return multi_entities
    """

class QA2D:
    def __init__(self):
        self.model_args = Seq2SeqArgs()
        self.model_args.max_length = 128
        self.model_args.eval_batch_size = 32
        self.QA2D_model = Seq2SeqModel(
                    encoder_decoder_type="bart", 
                    encoder_decoder_name="model_save/QA2D_model",
                    cuda_device=0,
                    args=self.model_args
                )

    def process_one(self, q, a):
        to_predict = [f"{q} [SEP] {a}"]
        try:
            results = self.QA2D_model.predict(to_predict)
        except:
            return None
        return results[0]

    def process_multi(self, qa):
        return_d = []
        to_predict = [f"{q} [SEP] {a}" for q, a in qa]
        try:
            results = self.QA2D_model.predict(to_predict)
        except:
            return None
        return results


class BoolQ2D:
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.VB_word_map = {"Did": "Does", "Will": "Would", "Shall": "Should", "May": "Might", "Had": "Has"}

    def process_one(self, boolq, a):
        from POSTree import POSTree
        boolq = boolq.strip()
        if boolq[-1] != '?':
            print("WARNING: not ending with question mark")
            print(boolq)
            return None
        subsents = boolq[:-1].split(',')
        if len(subsents) > 2:
            print("WARNING: subsents > 2")
            print(boolq)
            return None
        elif len(subsents) == 2:
            boolq = subsents[1].capitalize() + ' ' + self.decapitalize(subsents[0]) + '?'
        words = boolq.split(' ')
        words[0] = self.VB_word_map.get(words[0], words[0])
        boolq = ' '.join(words)
        try:
            output = self.nlp.annotate(boolq, properties={'annotators': 'parse', 'outputFormat': 'json'})
            parse_str = output['sentences'][0]['parse']
            tree = POSTree(parse_str)
            d = tree.adjust_order()
        except Exception as e:
            print("ERROR: ", e)
            print(boolq)
            return None
        if a:
            d = d.replace("**blank**", "")
        else:
            # TODO: add do/does/did not
            d = d.replace("**blank**", "not")
        d = d.replace('  ', ' ').capitalize()
        return d

    def process_multi(self, boolqa):
        return_d = []
        for boolq, a in tqdm(boolqa):
            d = self.process_one(boolq, a)
            return_d.append(d)
        return return_d

    def decapitalize(self, s):
        if len(s) == 0:
            return s
        elif len(s) == 1:
            return s.lower()
        else:
            s = s[0].lower() + s[1:]
            return s

def safe_replace(to_be_replaced, after_replaced, sent, count=0):
    try:
        pattern = r"([\s{}]){}([\s{}])".format(string.punctuation, to_be_replaced, string.punctuation)
        sent = ' ' + sent + ' '
        sent = re.sub(pattern, r'\g<1>{}\g<2>'.format(after_replaced), sent, count)
        sent = re.sub('\s+', ' ', sent)
        sent = sent.strip()
    except Exception as e:
        print(f"Replace Error: {to_be_replaced} -> {after_replaced} @@ {sent}")
    return sent

def safe_in(to_be_checked, sent):
    try:
        pattern = r"([\s{}]){}([\s{}])".format(string.punctuation, to_be_checked, string.punctuation)
        sent = ' ' + sent + ' '
        if re.search(pattern, sent):
            return True
    except Exception as e:
        print(f"Safe-in Error: {to_be_checked} @@ {sent}")
    return False


def shuffle_two_lists_together(a, b):
    assert len(a) == len(b)
    combined = list(zip(a, b))
    random.shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a, b

def keyword_extraction(sent):
    """
    r = Rake()
    r.extract_keywords_from_text(sent)
    return r.get_ranked_phrases()
    """
    """
    doc = nlp(sent)
    entities = [ent.text for ent in doc.ents]
    return entities
    """
    pass

def coref_resolution():
    pass

def extarct_nouns(sent):
    doc = nlp(sent)
    nouns = set()
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            nouns.add(token.text)
    return list(nouns)
