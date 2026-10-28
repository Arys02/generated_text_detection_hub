# %%

from itertools import combinations
import torch.utils.data
from hydra import initialize, compose
from omegaconf import DictConfig
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import similarity

from src.data.dataloader import data_loader
from src.data.dataset_aprdb import AprdbDataset
from sentence_transformers import SentenceTransformer
import src.config
from src.config import DATA_FOLDER
from src.detector.cristal_anchor.fetch_academic_paper import extract_text_from_pdf
import pandas as pd

def precompute_embeddings(
    dataframes: dict[str, pd.DataFrame],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Encode chaque corpus une seule fois. Retourne tensors sur GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = {}
    for name, df in dataframes.items():
        texts = df["content"].astype(str).tolist()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine = produit scalaire ensuite
        )
        embeddings[name] = emb
    return embeddings

def paired_mean_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Similarité cosinus moyenne sur les paires alignées (zip-like)."""
    n = min(emb_a.size(0), emb_b.size(0))
    # embeddings normalisés → dot product == cosine
    sims = (emb_a[:n] * emb_b[:n]).sum(dim=1)
    return sims.mean().item()

def main():
    with initialize(version_base=None, config_path="../conf"):
        cfg: DictConfig = compose(config_name="config", overrides=["+experiment/dataset/aprbd@dataset=aprdb_tt_test_all",
                                                                   "experiment/data_loader=1_4_true"])

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    print(cfg)

    # %%
    ds_test = AprdbDataset(config=cfg.dataset)
    len(ds_test)

    # %%
    import pandas as pd

    rows = []
    for i in range(len(ds_test)):
        item = ds_test[i]
        rows.append(item)
    df = pd.DataFrame(rows)

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df_generated = pd.read_json(src.config.DATA_DIR / 'reviews.json')
    df_generated = df_generated[['label', 'generated_review', 'years', 'conference']]
    df_generated['generator'] = "mgt"
    df_generated.rename(columns={'generated_review': 'content'}, inplace=True)

   # %%
    hwf_only_df = df[df['label'] == 0]
    mgt_only_df = df[df['label'] == 1]
    hwf_only_df = hwf_only_df.copy()
    # %%
    mgt_only_df.generator.unique()
    # %%
    claude_df = mgt_only_df[mgt_only_df['generator'] == 'claude']
    gpt4o_df = mgt_only_df[mgt_only_df['generator'] == 'gpt4o']
    gemini_df = mgt_only_df[mgt_only_df['generator'] == 'gemini']
    llama_df = mgt_only_df[mgt_only_df['generator'] == 'llama']
    qwen_df = mgt_only_df[mgt_only_df['generator'] == 'qwen']
    qwen_3_generated = df_generated[df_generated['generator'] == 'mgt']

    # %%
    hwf_only_df['generator'] = "human"
    # %%
    print("claude", claude_df.generator.unique())
    print("gemini", gemini_df.generator.unique())

    # %%

    corpora = {
        "human": hwf_only_df,
        "gemini": gemini_df,
        "llama": llama_df,
        "qwen": qwen_df,
        "gpt4o": gpt4o_df,
        "claude": claude_df,
        "qwen3": qwen_3_generated,
    }
    # 1. Encodage unique par corpus
    embeddings = precompute_embeddings(corpora, embedding_model, batch_size=256)
    results = {}
    for name_a, name_b in combinations(corpora.keys(), 2):
        results[(name_a, name_b)] = paired_mean_similarity(
            embeddings[name_a], embeddings[name_b]
        )

    import json
    # 3. Sérialisation
    data_clean = [{"a": a, "b": b, "value": v} for (a, b), v in results.items()]
    with open(src.config.DATA_DIR / f"similarities_{2}.json", "w", encoding="utf-8") as f:
        json.dump(data_clean, f, ensure_ascii=False, indent=2)
    return

    df_all_list = list(combinations([hwf_only_df, gemini_df, llama_df, qwen_df, gpt4o_df, claude_df, qwen_3_generated], 2))
    # %%
    def compare_similarity(a, b):
        embedded_a = embedding_model.encode(a, show_progress_bar=False)
        embedded_b = embedding_model.encode(b, show_progress_bar=False)
        similarity = embedding_model.similarity(embedded_a, embedded_b)
        return similarity

    # %%
    import tqdm

    values_similarity = {}

    for df_1, df_2 in tqdm.tqdm(df_all_list, total=len(df_all_list)):
        mean_similarities = 0
        count = 0
        print(df_1['generator'].unique(), df_2['generator'].unique())
        results = []
        for a, b in zip(df_1['content'], df_2['content']):
            results.append(compare_similarity(a, b))
        mean = sum(results) / len(results)
        values_similarity[(df_1['generator'].unique()[0], df_2['generator'].unique()[0]) ] = mean

    print(values_similarity)

    # %%
    import json

    def to_serializable(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return x

    data_clean = [
        {"a": a, "b": b, "value": v.item()}
        for (a, b), v in values_similarity.items()
    ]

    #infos_clean = [{(k, x): to_serializable(v) for x, v in d} for d in values_similarity]
    with open("similarities_2.json", "w", encoding="utf-8") as f:
        json.dump(data_clean, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
