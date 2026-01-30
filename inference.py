import argparse
import tqdm
from utils import *
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_reward

if __name__ == "__main__":
    seed_everything(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--split", default="test")
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    data_source = "lhkhiem28/TA-GRPO-datasets"
    print(f"Loading the {data_source}/{args.task} dataset from huggingface...", flush=True)
    dataset = load_dataset(
        data_source, args.task
    )
    dataset = dataset[args.split]
    dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
    generator = LLM(model=args.repo_id, tensor_parallel_size=1)
    embedder  = LLM(model="Qwen/Qwen3-Embedding-0.6B", tensor_parallel_size=1, task="embed")

    generator_embeddings = []
    generator_retvals = []
    for i in tqdm.tqdm(range(0, len(dataset), 128)):
        batch = dataset[i:i+128]
        batch_prompts = batch["prompt"]
        batch_prompts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in batch_prompts]

        batch_outputs = generator.generate(batch_prompts, SamplingParams(
            max_tokens=3072, n=8, 
            temperature=1.0, top_p=1, 
        ))
        batch_outputs = [[output.text for output in outputs.outputs] for outputs in batch_outputs]
        batch_embeddings  = torch.tensor([outputs.outputs.embedding for outputs in embedder.embed([output for outputs in batch_outputs for output in outputs])])
        batch_embeddings  = batch_embeddings.view(len(batch_prompts), 8, 1024).cpu().numpy()
        for embeddings in batch_embeddings:
            generator_embeddings.append(embeddings)
        for outputs, ground_truth in zip(batch_outputs, [item["ground_truth"] for item in batch["reward_model"]]):
            retvals = []
            for output in outputs:
                retval, _ = math_reward.compute_score(output, ground_truth)
                retvals.append(retval)
            generator_retvals.append(retvals)

    import os
    os.makedirs(f"outputs/{args.repo_id.split('/')[-1]}/{args.task}/{args.split}/", exist_ok=True)
    np.save(f"outputs/{args.repo_id.split('/')[-1]}/{args.task}/{args.split}/generator_embeddings.npy", np.array(generator_embeddings))
    np.save(f"outputs/{args.repo_id.split('/')[-1]}/{args.task}/{args.split}/generator_retvals.npy", np.array(generator_retvals))