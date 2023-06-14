import json
from glob import glob
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2')

sentences = []
all_files = ["data/preprocessed/omcs.json"]
outfile_name = "data/preprocessed/postprocessed/pairs_omcs.json"
# all_files = glob("data/preprocessed/statements/*.json")
# outfile_name = "data/preprocessed/postprocessed/pairs.json"
for file_name in all_files:
    with open(file_name, 'r') as reader:
        for json_line in reader.readlines():
            item = json.loads(json_line)
            sentences.append(item['statement'])
print(f"Total: {len(sentences)} sentences")

paraphrases = util.paraphrase_mining(model, sentences, show_progress_bar=True, batch_size=128)

print(outfile_name)
with open(outfile_name, 'w') as writer:
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        item = dict()
        item['statement_1'] = sentences[i]
        item['statement_2'] = sentences[j]
        item['score'] = score
        writer.write(json.dumps(item)+'\n')
