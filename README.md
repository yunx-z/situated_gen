# SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning

This repository contains the data and code for the baseline described in the following paper:

> [**SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning**](https://arxiv.org/pdf/2306.12552)<br/>
> Yunxiang Zhang, Xiaojun Wan<br/>
> Preprint. Under review.
```
@misc{zhang2023situatedgen,
      title={SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning}, 
      author={Yunxiang Zhang and Xiaojun Wan},
      year={2023},
      eprint={2306.12552},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Datasets

SituatedGen data files are located under `data/`.

- `train.jsonl` contains 5,641 training examples.
- `dev.jsonl` contains 1,407 development examples.
- `test.jsonl` contains 1,220 test examples.

The data files are formatted as jsonlines. Here is a single example:
```
{
      "keywords": ["approximately 365 days", "axis", "every 24 hours", "sun", "Earth", "Earth"],
      "statement": "Earth revolves around the sun approximately 365 days. Earth rotates on its axis once every 24 hours.",
      "ids": ["arc::Easy::Test::101", "arc::Easy::Test::110"],
      "keywords_pos": [0, 1, 1, 0, 1, 0],
      "statements": ["Earth revolves around the sun approximately 365 days.", "Earth rotates on its axis once every 24 hours."]
}
```

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `keywords`                | A list of input keywords                                                         |
| `statement`               | The target output, which is a string concatenation of two sentences containing these keywords                  |
| `ids`             | the origins of the two sentences (from which (train/dev/test) split of which source datasets/corpora) represented in the format of "\{src\_dataset\}::\{split\}::\{id\}"                               |
| `keywords_pos`             | At which sentence should the keyword appear (0 for the first sentence in "statements" field, 1 for the second)                                                                 |
| `statements`                | A list of the two sentences in the "statement" field                                                                              |


## Baselines
To train models (bart/t5/flan-t5), run the following command after you install the dependencies with `pip install -r requirements.txt` (Python=3.7)
```
bash scripts/keyword2text/train_plm.sh ${GPU_NO} ${MODEL_NAME}
```
To evaluate the predictions, run
```
bash scripts/keyword2text/pred_plm.sh ${GPU_NO} ${MODEL_NAME}
```
