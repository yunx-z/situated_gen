import json

for split in ["train", "dev"]:
    with open(f"data/raw/commongen/commongen.{split}.jsonl", 'r') as reader:
        items = [json.loads(line) for line in reader.readlines()]
    preprocessed_items = []
    for item in items:
        keywords = item['concept_set'].split('#')
        for scene in item['scene']:
            preprocessed_item = dict()
            preprocessed_item['keywords'] = keywords
            preprocessed_item['statement'] = scene
            preprocessed_items.append(preprocessed_item)
    with open(f"data/preprocessed/commongen/{split}.json", 'w') as writer:
        for item in preprocessed_items:
            writer.write(json.dumps(item)+'\n')
