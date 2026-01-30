import argparse
import tqdm
from utils import seed_everything, make_map_fn
from datasets import load_dataset

if __name__ == "__main__":
    seed_everything(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--split")
    parser.add_argument("--save_dir", default="../TA-GRPO-datasets")
    args = parser.parse_args()

    data_source = "lhkhiem28/TA-GRPO-datasets"
    print(f"Loading the {data_source}/{args.task} dataset from huggingface...", flush=True)
    dataset = load_dataset(
        data_source, args.task
    )

    dataset = dataset[args.split]
    dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

    import os
    local_dir = os.path.join(os.path.expanduser(args.save_dir), args.task)
    dataset.to_parquet(os.path.join(local_dir, f"{args.split}.parquet"))

    import json
    example = dataset[0]
    with open(os.path.join(local_dir, f"{args.split}_example.json"), "w") as f:
        json.dump(example, f, indent=2)