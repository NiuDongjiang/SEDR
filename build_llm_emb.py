import os
import json
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY = "sk-XXX"


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_api_key():
    return os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or DEFAULT_API_KEY


def make_client(base_url: Optional[str]):
    kwargs = {"api_key": resolve_api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def llm_encode(client, prompt: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a graph-structure semantic encoder. "
                    "Output only the encoded semantic text following the given format."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()


def llm_embed(client: OpenAI, text: str, emb_model: str):
    resp = client.embeddings.create(
        model=emb_model,
        input=text,
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def main(args):
    client = make_client(args.base_url)

    drug_prompts = load_prompts(os.path.join(args.data_dir, "drug_prompts.json"))
    disease_prompts = load_prompts(os.path.join(args.data_dir, "disease_prompts.json"))

    drug_embs = []
    disease_embs = []

    LOGGER.info("Encoding drug nodes...")
    for did in tqdm(drug_prompts):
        text = llm_encode(client, drug_prompts[did], args.llm_model)
        emb = llm_embed(client, text, args.emb_model)
        drug_embs.append(emb)
        time.sleep(args.sleep)

    LOGGER.info("Encoding disease nodes...")
    for did in tqdm(disease_prompts):
        text = llm_encode(client, disease_prompts[did], args.llm_model)
        emb = llm_embed(client, text, args.emb_model)
        disease_embs.append(emb)
        time.sleep(args.sleep)

    np.save(os.path.join(args.data_dir, "drug_llm_emb.npy"), np.vstack(drug_embs))
    np.save(os.path.join(args.data_dir, "disease_llm_emb.npy"), np.vstack(disease_embs))

    LOGGER.info("LLM graph semantic embedding finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Gdataset')
    parser.add_argument("--llm_model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--emb_model", default="text-embedding-v4")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sleep", type=float, default=0.3)

    args = parser.parse_args()
    main(args)
