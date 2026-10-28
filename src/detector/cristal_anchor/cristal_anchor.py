import re
from typing import Union

import numpy as np
import torch
from openai import OpenAI
from torch import tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from src.detector.base import BaseDetector
import os

from src.detector.cristal_anchor.embedding_model import embbeding_model_factory
from src.detector.detector_utils import huggingface_config, from_pretrained

from src.detector.cristal_anchor.fetch_academic_paper import extract_text_from_pdf
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.error, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MessageList = list[dict[str, str]]
PromptInput = Union[str, MessageList, list[str], list[MessageList]]
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
_client = OpenAI(base_url=VLLM_API_BASE, api_key="EMPTY")

REVIEW_OUTPUT_SPEC = """

Return your answer in the following exact structure, using these literal markers.

The reconstructed prompts MUST be self-contained: a third party reading only
<RECONSTRUCTED_SYSTEM_PROMPT> + <RECONSTRUCTED_USER_PROMPT> + an arbitrary paper
must be able to regenerate a text isomorphic to the target.

<RECONSTRUCTED_SYSTEM_PROMPT>
Must crystallize the following invariants from Phase 1/2:
- PERSONA: assigned role, expertise level, viewpoint (I2)
- REGISTER & STYLE: lexical field, hedging frequency, syntactic patterns (I3, I4)
- FORMATTING RULES: Markdown usage, citation style, typography (I5)
- TONE CONSTRAINTS: imposed stance, modality (I7)
- LLM-specific stylistic directives (I8) if detected

Write as a coherent role description, NOT as a bullet list of invariants.
Do NOT mention the paper, sections, or task-specific instructions here.
</RECONSTRUCTED_SYSTEM_PROMPT>

<RECONSTRUCTED_USER_PROMPT>
Must crystallize:
- TASK STATEMENT: the direct instruction inferred in Phase 2
- MANDATORY SECTIONS with ordering (I1) — list them explicitly
- EVALUATION CRITERIA with their weighting (I6)
- LENGTH DIRECTIVE: explicit word target based on the analyse of the based review(I9)
- The literal placeholder {paper} where the input paper must be injected

Write as an executable instruction. End with the {paper} block.
</RECONSTRUCTED_USER_PROMPT>

<MAPPING_TABLE>
Two-column table showing where each Phase 1 invariant (I1..I9) and each Phase 2
deduction was placed (SYSTEM | USER | OMITTED) with a one-line justification.
Any invariant marked OMITTED must justify why it was not actionable.
</MAPPING_TABLE>

<JUSTIFICATION>
For each non-trivial choice in SYSTEM and USER prompts, cite the textual evidence
(short quote from target) that motivates it. Format: "[quote] → [prompt element]".
</JUSTIFICATION>

<FALSIFICATION>
3 alternative reconstructions and why they are rejected.
Residual ambiguities requiring empirical iteration.
</FALSIFICATION>"""


class CristalAnchor(BaseDetector):
    def __init__(self, config):
        super(CristalAnchor, self).__init__(config)
        self.params = config['params']
        self.cache_dir = os.getenv('CACHE_DIR')

        self.params_generate_review = config['params_generate_review']
        self.params_reverse_prompt = config['params_reverse_prompt']
        self.embedding_model_name = self.params['embedder_name']


        ### EMBEDDING ###
        self.embedding_model = embbeding_model_factory(self.embedding_model_name)

    @staticmethod
    def build_prompt(txt, review_spec):
        return f"""# ROLE
    You are an expert in reverse prompt engineering, specialized in retro-engineering prompts from LLM outputs. Your goal: reconstruct the prompt that most likely generated a given target text, such that re-executing this prompt produces an output stylistically and structurally isomorphic to the target.


    # INPUT
    <TARGET_TEXT>
    {txt}
    </TARGET_TEXT>

    # METHODOLOGY (execute explicitly, step by step)

    ## Phase 1 — Forensic analysis of the target text
    Extract and document the following invariants:

    1. **Macro-structure**: sections, ordering, hierarchy, transitions, relative length of each block.
    2. **Implicit persona**: displayed expertise level, stance (critical/supportive/neutral), domain, viewpoint (NeurIPS/ICML/ACL reviewer? editor? naive reader?).
    3. **Lexical register**: dominant semantic field, technical terminology, frequency of hedges ("arguably", "it seems"), recurring logical connectors.
    4. **Syntactic patterns**: average sentence length, passive/active voice ratio, first-person usage, modalities.
    5. **Formatting conventions**: Markdown usage, presence of lists, citations, formulas, typography (italics for technical terms?).
    6. **Evaluation axes**: which criteria does the text judge on (originality, soundness, clarity, significance, reproducibility)? With what apparent weighting?
    7. **Implicit constraints**: length limits, mandatory sections, imposed tone (cf. conference reviewer guidelines such as NeurIPS).
    8. **LLM-specific signals** (if relevant): tell-tale phrasings of specific models (GPT-4, Claude, Gemini), stylistic tics, canonical formulas.
    9. **size** : the size is close to {len(str(txt).split())} words long.

    ## Phase 2 — Inference of latent parameters
    Deduce:
    - The probable **system prompt** (assigned role, format constraints).
    - The probable **user prompt** (direct instruction, provided context, requirements, size of the target based on the text).
    - Possible **few-shot examples** (does the text follow a visible template?).
    - Plausible **generation parameters** (low/high temperature based on observed creativity).

    ## Phase 3 — Synthesis of the candidate prompt
    Produce the reconstructed prompt respecting:
    - The **language** of the target text (English).
    - A directly executable format (system + user).
    - A specification sufficiently **constraining** to reproduce the structure, yet sufficiently **generalizable** to work on a different paper as input.
    - Explicit inclusion of sections, tone, criteria, and any detected length constraints.

    ## Phase 4 — Line-by-line justification
    For each element of the reconstructed prompt, indicate as a comment the textual evidence (short quote from the target text) motivating that choice.

    ## Phase 5 — Falsification test
    List 3 alternative hypotheses about the source prompt and explain why you reject them. Identify residual ambiguities that only empirical iteration could resolve.

    # OUTPUT (strict format)""" + review_spec

    def chat(self, messages, *, enable_thinking=False, temperature=0.7, top_p=0.95,
             top_k=20, max_tokens=2048):
        if enable_thinking and temperature > 0.6:
            temperature = 0.6
        resp = _client.chat.completions.create(
            model=self.params["model_name"],
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "top_k": top_k,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        msg = resp.choices[0].message
        return {
            "content": (msg.content or "").strip(),
            "reasoning": getattr(msg, "reasoning_content", None),
            "finish_reason": resp.choices[0].finish_reason,
            "usage": dict(resp.usage),
        }
    @staticmethod
    def _extract_tag(text: str, tag: str, required: bool = True) -> str | None:
        """Extrait le contenu d'un tag. Si non requis et absent, renvoie None."""
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
        if not m:
            # Fallback : tag ouvert mais pas refermé (sortie tronquée)
            m_open = re.search(rf"<{tag}>(.*)", text, flags=re.DOTALL)
            if m_open and not required:
                return m_open.group(1).strip()
            if required:
                raise ValueError(f"Required tag <{tag}> not found in output")
            return None
        return m.group(1).strip()

    def reverse_engineer_prompt(self, review_text: str,
                                enable_thinking: bool = True,
                                temperature: float = 0.6,
                                max_token: int = 20000) -> dict:
        """Étape 1 : extrait system_prompt + user_prompt depuis une review."""
        out = self.chat(
            [{"role": "user", "content": self.build_prompt(review_text, REVIEW_OUTPUT_SPEC)}],
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_token,
        )
        raw = out["content"]

        if out.get("finish_reason") == "length":
            print(f"[WARN] Reverse-engineering output truncated "
                  f"(usage: {out['usage']})")
        return {
            "system_prompt": self._extract_tag(raw, "RECONSTRUCTED_SYSTEM_PROMPT"),
            "user_prompt_template": self._extract_tag(raw, "RECONSTRUCTED_USER_PROMPT"),
            "justification": self._extract_tag(raw, "JUSTIFICATION", required=False),
            "falsification": self._extract_tag(raw, "FALSIFICATION", required=False),
            "raw": raw,
            "usage": out["usage"],
            "reasoning": out["reasoning"],
        }

    def generate_review(self, paper: str, reconstructed: dict,
                        enable_thinking: bool = False,
                        temperature: float = 0.7,
                        max_tokens: int = 4096,
                        top_p: float = 0.95,
                        top_k: int = 20) -> dict:
        """Étape 2 : applique le prompt reconstruit au papier cible."""
        if "{paper}" not in reconstructed["user_prompt_template"]:
            # fallback si le modèle a oublié le placeholder
            user_content = reconstructed["user_prompt_template"] + f"\n\n<PAPER>\n{paper}\n</PAPER>"
        else:
            user_content = reconstructed["user_prompt_template"].replace("{paper}", paper)

        return self.chat(
            [
                {"role": "system", "content": reconstructed["system_prompt"]},
                {"role": "user", "content": user_content},
            ],
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )

    def review_pipeline(self, review_ref: str, paper_text: str) -> dict:
        reconstructed = self.reverse_engineer_prompt(review_ref,
                                                     temperature = self.params_reverse_prompt['temperature'],
                                                     enable_thinking = self.params_reverse_prompt['enable_thinking'],
                                                     max_token=self.params_reverse_prompt['max_token'])

        generated = self.generate_review(paper_text, reconstructed,
                                         max_tokens=self.params_generate_review['max_token'],
                                         enable_thinking=self.params_generate_review['enable_thinking'],
                                         top_p=self.params_generate_review['top_p'],
                                         top_k=self.params_generate_review['top_k'],
                                         temperature=self.params_generate_review['temperature'])
        return {
            "reconstructed": reconstructed,
            "generated_review": generated["content"],
            "generated_reasoning": generated["reasoning"],
            "usage": {
                "reverse": reconstructed["usage"],
                "generate": generated["usage"],
            },
        }

    def predict_(self, batch):
        texts = batch['content']
        links = batch['pdf_path']
        if isinstance(texts, str):
            texts = [texts]

        scores = []
        for link, review in zip(links, texts):
            try:
                paper = extract_text_from_pdf(link)
                r = self.review_pipeline(review, paper)
                gen = r['generated_review']
                if not review.strip() or not gen.strip():
                    raise ValueError("review ou génération vide")
                # similarité diagonale : review_i vs sa reconstruction_i
                sim = self.embedding_model.compare(review, gen)  # (1,1)
                scores.append(float(np.asarray(sim).reshape(-1)[0]))
            except Exception as e:
                logger.error(f"predict item échoué: {e}")
                scores.append(float("nan"))  # place tenue -> alignement préservé

        return np.array(scores, dtype=np.float64).reshape(-1, 1)

    def predict(self, batch):
        texts = batch['content']
        links = batch['pdf_path']

        if isinstance(texts, str):
            texts = [texts]


        results = []
        txts = []
        reconstructured = []
        for link, review in zip(links, texts):
            paper = extract_text_from_pdf(link)
            # try:
            r = self.review_pipeline(review, paper)
            # except Exception as e:
            #     logger.error(e)
            #     print(e)
            reconstructured.append(r)
            txts.append(review)
            results.append(r['generated_review'])


        embedded_text = self.embedding_model.encode(txts)
        embedded_results = self.embedding_model.encode(results)

        similarities = self.embedding_model.compare(texts, results)
        print(similarities)
        return similarities


if __name__ == "__main__":
    config = {
        "params": {
            "model_name": "qwen3-14b",
            "seed": 42
        },
        "params_generate_review": {
            "temperature": 0.7,
            "max_token": 4096,
            "top_p": 0.95,
            "top_k": 20,
            "enable_thinking": False,
        },
        "params_revers_prompt": {
            "temperature": 0.6,
            "enable_thinking": True,
            "max_tokens": 20000,
        }
    }
    random_batch = {'label': tensor([1]), 'content': ["The paper introduces the polynomial convolutional neural network (PolyCNN), a variant of the traditional CNN designed to reduce the number of learnable parameters while maintaining performance. The core idea is to use a single 'seed' filter at each convolutional layer, which is then transformed into multiple filters using polynomial operations. This approach, termed 'early fan-out' and 'late fan-out', aims to reduce model complexity and improve efficiency.\n\n**Strengths:*\n1. **Novelty**: The concept of using polynomial transformations to generate multiple filters from a single seed filter is an interesting and novel idea. It aligns with the broader goal of making CNNs more efficient.\n2. **Efficiency**: The paper demonstrates significant parameter savings, ranging from 10× to 50×, which is a substantial improvement in terms of model complexity.\n3. **Comprehensive Experiments**: The authors have conducted experiments on multiple datasets (MNIST, SVHN, CIFAR-10, and ImageNet) and compared the proposed PolyCNN with standard CNNs and other state-of-the-art methods.\n\n**Weaknesses:*\n1. **Performance**: While the paper claims on-par performance, the results on several benchmarks show that PolyCNN often underperforms compared to standard CNNs and other state-of-the-art methods. For example, on CIFAR-10, the best PolyCNN (early fan-out) achieves 90.56% accuracy, which is significantly lower than the baseline (92.95%) and other methods like Maxout (90.65%) and NIN (91.19%).\n2. **Lack of Theoretical Justification**: The paper lacks a strong theoretical foundation for why polynomial transformations should work well in this context. While the authors draw inspiration from polynomial correlation filters, the connection is not thoroughly explored or justified.\n3. **Hyperparameter Sensitivity**: The performance of PolyCNN seems to be highly sensitive to hyperparameters, such as the number of seed filters and the type of fan-out. This sensitivity might limit its practical applicability.\n4. **Clarity of Presentation**: The paper is somewhat dense and convoluted, which makes it difficult to follow. The notation and equations are not always clear, and some sections (like the derivations in Section 2) could benefit from more intuitive explanations.\n5. **Comparative Analysis**: The comparison with other methods is not as comprehensive as it could be. For instance, the paper does not discuss recent advances in efficient CNNs, such as mobile architectures (e.g., MobileNet, EfficientNet), which are also designed to reduce model complexity.\n\n**Conclusion**: The idea of PolyCNN is intriguing and has the potential to contribute to the field of efficient deep learning. However, the current implementation and experimental results do not yet demonstrate the claimed on-par performance, and the paper lacks a strong theoretical foundation. The presentation and clarity of the paper could also be improved. Given these issues, the paper is not yet ready for acceptance and should be rejected. The authors are encouraged to address the performance gaps, provide a stronger theoretical justification, and improve the clarity of the presentation before resubmitting."], 'subset': ['test'], 'generator': ['qwen'], 'conference': ['ICLR'], 'PaperId': tensor([1170]), 'year': ['2019'], 'file': ['test/qwen/ICLR2019.test.qwen2.5-72b.csv'], 'review-id': ['Hye_oQI6nX'], 'PaperIdFull': ['ICLR-2019-1170'], 'forum_id': ['B1GHb2RqYX'], 'pdf_path': ['/path/to/cache_dir/papers_pdf/B1GHb2RqYX.pdf']}

    from src.detector.cristal_anchor.fetch_academic_paper import extract_text_from_pdf

    r = random_batch["content"][0] if isinstance(random_batch["content"], list) else random_batch["content"]
    p = extract_text_from_pdf(random_batch["pdf_path"][0])

    detector = CristalAnchor(config)

    result = detector.review_pipeline(r, p)

    print(result)


    print("=== RECONSTRUCTED SYSTEM ===")
    print(result["reconstructed"]["system_prompt"])
    print("\n=== RECONSTRUCTED USER (template) ===")
    print(result["reconstructed"]["user_prompt_template"])
    print("\n=== GENERATED REVIEW ===")
    print(result["generated_review"], len(result["generated_review"]))
    print("\n=== REASONING ===")
    print(result["reconstructed"]["reasoning"])
    print("\n=== GENERATED REVIEW ===")
    print("\n=== USAGE ===")
    print(result["usage"])
    result_review = result['generated_review']


    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embedded_target = embedding_model.encode(random_batch["content"][0])
    embedded_generated_review = embedding_model.encode(result_review)

    similarity = embedding_model.similarity(embedded_target, embedded_generated_review)
    print(similarity)
