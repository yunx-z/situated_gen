import argparse                                                                                                                    

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", help="multi dataset names", nargs='*')
args = parser.parse_args()
datasets = args.datasets
if datasets is None:
    datasets = ["creak", "strategyqa", "commonsenseqa", "arc", "openbookqa", "omcs", "qasc"]

for dataset in datasets:
    if dataset == "creak":
        from dataset_preprocessor.FV import CREAKPreprocessor
        preprocessor = CREAKPreprocessor()
    elif dataset == "strategyqa":
        from dataset_preprocessor.BoolQA import StrategyQAPreprocessor
        preprocessor = StrategyQAPreprocessor()
    elif dataset == "commonsenseqa":
        from dataset_preprocessor.MCQA import CommonsenseQAPreprocessor
        preprocessor = CommonsenseQAPreprocessor()
    elif dataset == "arc":
        from dataset_preprocessor.MCQA import ARCPreprocessor
        preprocessor = ARCPreprocessor()
    elif dataset == "openbookqa":
        from dataset_preprocessor.FV import OpenbookQAPreprocessor
        preprocessor = OpenbookQAPreprocessor()
    elif dataset == "omcs":
        from dataset_preprocessor.FV import OMCSPreprocessor
        preprocessor = OMCSPreprocessor()
    elif dataset == "qasc":
        from dataset_preprocessor.FV import QASCPreprocessor
        preprocessor = QASCPreprocessor()
    else:
        raise ValueError("invalid dataset name!")
    print(f"preprocessing {dataset} ...")
    preprocessor.run()
