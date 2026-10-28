from typing import List, Union

from transformers import AutoTokenizer
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PlotData:
    values: Union[List[float] | List[int]]
    label: str


def display_percentile(list_plot_data: List[PlotData]):
    x = np.arange(len(list_plot_data[0].values))
    plt.figure()
    for i, p in enumerate(list_plot_data):
        plt.plot(x, p.values, label=f"label {i}  ({p.label})")
    plt.xlabel("Percentile")
    plt.ylabel("Longueur (mot/token)")
    plt.title("Courbes de percentiles des longueurs (0 vs 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def display_histogram(list_plot_data: List[PlotData]):

    xmin = min(min(d.values) for d in list_plot_data)
    xmax = max(max(d.values) for d in list_plot_data)

    bins = np.linspace(xmin, xmax, int(np.sqrt(len(list_plot_data[0].values))))  # ajuste 60 si besoin

    plt.figure()
    for p in list_plot_data:
        plt.hist(p.values, bins=bins, alpha=0.5, label=p.label, orientation='horizontal')

    plt.xlabel("Length (words)")
    plt.ylabel("Count")
    plt.title("Histogrammes superposés des longueurs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def basic_audit(dataset, max_items=None, tokenized=False):
    labels = []
    lengths = []
    flags = Counter()
    texts = []
    datas_for_label = {
        0: {
            "empty": 0,
            "url": 0,
            "bullet": 0,
            "multi_newlines": 0,
            "bracket_refs": 0,
            "length_content": []
        },
        1: {
            "empty": 0,
            "url": 0,
            "bullet": 0,
            "multi_newlines": 0,
            "bracket_refs": 0,
            "length_content": []
        }
    }

    if not tokenized:
        url_re = re.compile(r"https?://|www\.")
        bullet_re = re.compile(r"(^|\n)\s*([-•*]|\d+\)|\d+\.)\s+")
        multi_nl_re = re.compile(r"\n{3,}")
        bracket_ref_re = re.compile(r"\[\d+\]")

    n = len(dataset) if max_items is None else min(len(dataset), max_items)

    for i in range(n):
        item = dataset[i]
        y = int(item["label"])
        x = item["content"] or ""

        if not tokenized:
            datas_for_label[y]["empty"] += int(len(x) == 0)
            datas_for_label[y]["url"] += int(bool(url_re.search(x)))
            datas_for_label[y]["bullet"] += int(bool(bullet_re.search(x)))
            datas_for_label[y]["multi_newlines"] += int(bool(multi_nl_re.search(x)))
            datas_for_label[y]["bracket_refs"] += int(bool(bracket_ref_re.search(x)))
        datas_for_label[y]["length_content"].append(len(x))


        labels.append(y)
        texts.append(x)
        lengths.append(len(x))

        if not tokenized:
            flags["empty"] += int(len(x) == 0)
            flags["url"] += int(bool(url_re.search(x)))
            flags["bullets"] += int(bool(bullet_re.search(x)))
            flags["multi_newlines"] += int(bool(multi_nl_re.search(x)))
            flags["bracket_refs"] += int(bool(bracket_ref_re.search(x)))

    label_counts = Counter(labels)

    # exact duplicates
    # dup_count = n - len(set(texts))

    def summarize(arr):
        arr = np.array(arr)
        return {
            "p": [float(np.percentile(arr, i)) for i in range(0, 99)],
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    # lengths by class
    lengths_by_class = {}
    for cls in sorted(label_counts):
        cls_lengths = [l for l, y in zip(lengths, labels) if y == cls]
        lengths_by_class[cls] = summarize(cls_lengths)

    return {
        "N": n,
        "label_counts": dict(label_counts),
        "flags_for_label": datas_for_label,
        "lengths_by_class_words": lengths_by_class,
        "flags": dict(flags),
        # "exact_duplicate_count": dup_count,
    }


def basic_audit_tokenized(dataset, max_items=None, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
    labels = []
    lengths = []
    flags = Counter()
    texts = []
    datas_for_label = {
        0: {
            "empty": 0,
            "url": 0,
            "bullet": 0,
            "multi_newlines": 0,
            "bracket_refs": 0,
            "length_content": []
        },
        1: {
            "empty": 0,
            "url": 0,
            "bullet": 0,
            "multi_newlines": 0,
            "bracket_refs": 0,
            "length_content": []
        }
    }

    url_re = re.compile(r"https?://|www\.")
    bullet_re = re.compile(r"(^|\n)\s*([-•*]|\d+\)|\d+\.)\s+")
    multi_nl_re = re.compile(r"\n{3,}")
    bracket_ref_re = re.compile(r"\[\d+\]")

    n = len(dataset) if max_items is None else min(len(dataset), max_items)

    for i in range(n):
        item = dataset[i]
        y = int(item["label"])
        x = item["content"] or ""
        x = x.strip()

        datas_for_label[y]["empty"] += int(len(x) == 0)
        datas_for_label[y]["url"] += int(bool(url_re.search(x)))
        datas_for_label[y]["bullet"] += int(bool(bullet_re.search(x)))
        datas_for_label[y]["multi_newlines"] += int(bool(multi_nl_re.search(x)))
        datas_for_label[y]["bracket_refs"] += int(bool(bracket_ref_re.search(x)))

        # Tokenizing the content
        tokens = tokenizer.encode(x, add_special_tokens=False)
        datas_for_label[y]["length_content"].append(len(tokens))

        labels.append(y)
        texts.append(x)
        lengths.append(len(tokens))

        flags["empty"] += int(len(x) == 0)
        flags["url"] += int(bool(url_re.search(x)))
        flags["bullets"] += int(bool(bullet_re.search(x)))
        flags["multi_newlines"] += int(bool(multi_nl_re.search(x)))
        flags["bracket_refs"] += int(bool(bracket_ref_re.search(x)))

    label_counts = Counter(labels)

    # exact duplicates
    dup_count = n - len(set(texts))

    def summarize(arr):
        arr = np.array(arr)
        return {
            "p": [float(np.percentile(arr, i)) for i in range(0, 99)],
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    # lengths by class
    lengths_by_class = {}
    for cls in sorted(label_counts):
        cls_lengths = [l for l, y in zip(lengths, labels) if y == cls]
        lengths_by_class[cls] = summarize(cls_lengths)

    return {
        "N": n,
        "label_counts": dict(label_counts),
        "flags_for_label": datas_for_label,
        "lengths_by_class_tokens": lengths_by_class,
        "flags": dict(flags),
        "exact_duplicate_count": dup_count,
    }
