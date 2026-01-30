import argparse
import tqdm
from utils import *
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_reward

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

if __name__ == "__main__":
    seed_everything(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--split", default="test")
    parser.add_argument("--folder_path")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    if args.folder_path is not None:
        from huggingface_hub import HfApi, upload_folder
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        upload_folder(
            repo_id=args.repo_id, repo_type="model", 
            folder_path=args.folder_path
        )
    else:
        data_source = "lhkhiem28/TA-GRPO-datasets"
        print(f"Loading the {data_source}/{args.task} dataset from huggingface...", flush=True)
        dataset = load_dataset(
            data_source, args.task
        )
        dataset = dataset[args.split]
        dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

        generator = LLM(model=args.repo_id, tensor_parallel_size=1)

        pass_1, pass_8, pass_16, pass_32 = [], [], [], []
        for i in tqdm.tqdm(range(0, len(dataset), 128)):
            batch = dataset[i:i+128]
            batch_prompts = [item[0]["content"] for item in batch["prompt"]]

            batch_outputs = generator.generate(batch_prompts, SamplingParams(
                max_tokens=3072, n=32, 
                temperature=0.6, top_p=0.95, 
            ))
            batch_outputs = [[output.text for output in outputs.outputs] for outputs in batch_outputs]
            for outputs, ground_truth in zip(batch_outputs, [item["ground_truth"] for item in batch["reward_model"]]):
                retvals = []
                for output in outputs:
                    retval, _ = math_reward.compute_score(output, ground_truth)
                    retvals.append(retval)
                pass_1.append(pass_at_k(32, sum(retvals), 1))
                pass_8.append(pass_at_k(32, sum(retvals), 8))
                pass_16.append(pass_at_k(32, sum(retvals), 16))
                pass_32.append(pass_at_k(32, sum(retvals), 32))

        print("pass@1: {:.3f}".format(100*sum(pass_1)/len(pass_1)))
        print("pass@8: {:.3f}".format(100*sum(pass_8)/len(pass_8)))
        print("pass@16: {:.3f}".format(100*sum(pass_16)/len(pass_16)))
        print("pass@32: {:.3f}".format(100*sum(pass_32)/len(pass_32)))