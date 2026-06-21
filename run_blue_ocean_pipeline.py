"""
Blue Ocean Pipeline v3.0 — PhD-Defense-Ready Analysis
ODE Protocol compliant. No synthetic data. All metrics grounded in corpus.
"""
import gc
import os
import sys
import json
import argparse
import time
import re
import pickle
from collections import Counter, defaultdict
from itertools import chain

# Repository root (https://github.com/Aslmalki/NLP-to-Modeling-pipeline)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

_env_paths = [os.path.join(PROJECT_ROOT, ".env")]


def _load_env():
    """Load .env into os.environ. Uses dotenv if available, else manual parse."""
    for p in _env_paths:
        if not os.path.exists(p):
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(p)
            return
        except ImportError:
            pass
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    v = v.strip().strip('"').strip("'")
                    os.environ[k.strip()] = v
        return


_load_env()

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

from config import (
    set_all_seeds,
    REPRODUCIBILITY_SEED,
    DOC2VEC_SEED,
    FIGURES_DIR,
    VALIDATION_DIR,
    TOPIC_ASSIGNMENTS_DIR,
    HUMAN_ANNOTATION_CSV,
    INPUT_FOLDER,
    CODEBOOK_CSV,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_SEED,
    ENABLE_GEMINI_FALLBACK,
    VALIDATION_SUBSET_IDS_PATH,
    load_validation_subset_ids,
)
from src.preprocessing import get_custom_stopwords, load_data_from_drive, preprocess_text
from src.embeddings import get_doc2vec_embedding, get_bert_embedding, get_tfidf_embedding
from bertopic import BERTopic
import umap

# Validation artifacts → outputs/validation/; topic assignments → outputs/topic_assignments/
RESULTS_DIR = VALIDATION_DIR
CSV_PATH = HUMAN_ANNOTATION_CSV
BERTOPIC_STATE_PATH = os.path.join(RESULTS_DIR, "bertopic_validation_state.pkl")
# Frozen 78-document corpus fit (no Palinkas); use with --extend-validation
BERTOPIC_STATE_78DOC_PATH = os.path.join(RESULTS_DIR, "bertopic_validation_state_78doc.pkl")
UMAP_COORDINATES_CSV = os.path.join(RESULTS_DIR, "umap_coordinates.csv")
TOPIC_LABELS_CSV = os.path.join(RESULTS_DIR, "topic_labels.csv")
# Paper codebook labels and validation-subset document counts (frozen reporting values).
PAPER_TOPIC_LABEL_TABLE = [
    (0, "Space Mission & Crew Team Dynamics", 37),
    (1, "Team Agent Modeling & Desert Environments", 10),
    (2, "Climate Change & Agricultural Temperature", 3),
    (3, "Cold Exposure & Performance", 8),
    (4, "Offshore Work & Occupational Health", 5),
]
UMAP_SCATTER_FIG_PNG = os.path.join(FIGURES_DIR, "umap_bertopic_topics_publication.png")
UMAP_SCATTER_FIG_PDF = os.path.join(FIGURES_DIR, "umap_bertopic_topics_publication.pdf")
# Figure colors (match publication-style UMAP: blue / green / red / gold; outliers neutral)
BERTOPIC_UMAP_FIG_COLORS = {
    -1: "#888888",
    0: "#1a56b0",
    1: "#2d8a54",
    2: "#c42b2b",
    3: "#c9920d",
    4: "#6b3fa0",
}
VALIDATED_RUN_UMAP_ALL108 = os.path.join(RESULTS_DIR, "validated_run_umap_all108.csv")
VALIDATED_RUN_UMAP_63 = os.path.join(RESULTS_DIR, "validated_run_umap_63.csv")
VALIDATED_UMAP_PNG = os.path.join(FIGURES_DIR, "validated_UMAP.png")
EXPECTED_VALIDATED_EXPERT_COUNTS = {0: 37, 1: 10, 2: 3, 3: 8, 4: 5}
EXPECTED_VALIDATED_N = 63
VALIDATED_SUB_UMAP_COLORS = {
    -1: "#6B7280",
    0: "#2563EB",
    1: "#16A34A",
    2: "#DC2626",
    3: "#D97706",
    4: "#7C3AED",
}
VALIDATED_UMAP_CAPTION = (
    "UMAP projection of the 63 validated documents, colored by BERTopic "
    "topic assignment. Embeddings were computed on the full 108-document "
    "corpus; this subset shows only documents with expert annotations "
    "used for validation."
)
# Papers that must use Doc2Vec infer_vector + BERTopic.transform (not training-time dv lookup)
EXTEND_FORCE_INFER_PAPER_IDS = frozenset({
    "DAntoine_2023_Psychosocial",
    "Pawlak_2023_Lost",
    "Bzdok_2022_Social",
    "Palinkas_2021_Psychosocial",
})


def load_topic_definitions(filepath=None):
    """
    Loads the human-validated topic codebook from an external CSV file.
    This decouples the interpretive definitions from the pipeline logic,
    making the pipeline reusable on any new dataset by simply updating the CSV.
    """
    filepath = filepath or CODEBOOK_CSV
    try:
        df = pd.read_csv(filepath)
        return pd.Series(df.definition.values, index=df.topic_name).to_dict()
    except FileNotFoundError:
        print(f"FATAL: The codebook file was not found at '{filepath}'.")
        print("Please create the file and populate it with your topic definitions before running.")
        sys.exit(1)


def build_runtime_topic_definitions(model, topic_names_dict, static_csv_path):
    """
    Codebook for this pipeline run: use human definitions from CSV when the BERTopic
    topic name matches; otherwise define each category from this run's c-TF-IDF top terms.
    """
    static = {}
    if static_csv_path and os.path.isfile(static_csv_path):
        try:
            df = pd.read_csv(static_csv_path)
            static = pd.Series(df.definition.values, index=df.topic_name.astype(str)).to_dict()
        except Exception:
            static = {}
    definitions = {}
    for tid in sorted(k for k in topic_names_dict.keys() if k != -1):
        name = str(topic_names_dict[tid])
        if name in static:
            definitions[name] = static[name]
            continue
        tw = model.get_topic(tid)
        top_words = ", ".join([w for w, _ in (tw or [])[:30]])
        definitions[name] = (
            f"BERTopic cluster from this run (internal id {tid}). "
            f"Key lexical terms extracted by the model: {top_words}."
        )
    return definitions


def _extract_paper_id_from_filename(filename):
    """Extract paper ID from filename like 'Bishop_2009_FMARS_output.txt' -> 'Bishop_2009_FMARS'."""
    base = os.path.basename(filename)
    if base.endswith("_output.txt"):
        return base[:-11]  # remove _output.txt
    if base.endswith(".txt"):
        return base[:-4]
    return base


def _normalize_paper_id(pid):
    """Normalize for matching (e.g., handle typos like Pyschosocial vs Psychosocial)."""
    if pd.isna(pid) or not str(pid).strip():
        return None
    return str(pid).strip()


def _match_csv_row_to_doc(csv_df, filenames):
    """Build mapping: csv_row_index -> (doc_index, paper_id)."""
    doc_id_to_idx = {}
    for i, fn in enumerate(filenames):
        pid = _extract_paper_id_from_filename(fn)
        doc_id_to_idx[pid] = i
        # Also try without last word (e.g., Psychosocial vs Pyschosocial)
        parts = pid.rsplit("_", 1)
        if len(parts) == 2:
            doc_id_to_idx[parts[0]] = i  # allow prefix match as fallback

    id_col = None
    for c in csv_df.columns:
        if "Paper ID" in c or c == "Paper ID (LastNameFirstAuthor_Year_FirstWordTitle)":
            id_col = c
            break
    if id_col is None:
        id_col = csv_df.columns[0]

    mapping = {}
    for idx, row in csv_df.iterrows():
        pid = _normalize_paper_id(row.get(id_col, ""))
        if not pid:
            mapping[idx] = (None, None)
            continue
        doc_idx = doc_id_to_idx.get(pid)
        if doc_idx is None:
            # Try fuzzy: e.g. Parkes_1998_Pyschosocial vs Parkes_1998_Psychosocial
            for k, v in doc_id_to_idx.items():
                if k and pid and (k.startswith(pid.split("_")[0]) or pid.startswith(k.split("_")[0])):
                    if len(pid) > 10 and len(k) > 10 and abs(len(pid) - len(k)) <= 2:
                        doc_idx = v
                        break
        mapping[idx] = (doc_idx, pid)
    return mapping


def _canonical_topic_id_from_label(label):
    """
    Map BERTopic/LLM topic strings to a single integer codebook id (prefix before first '_').
    Examples: '0_team_agent_task_norm' and '0_team_agent_task_model' -> 0.
    Outliers / unassigned -> -1. Returns None if the label cannot be parsed.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    s = str(label).strip()
    sl = s.lower()
    if "outlier" in sl or "unassigned" in sl:
        return -1
    if "_" not in s:
        return None
    prefix = s.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def _canonical_topic_id_bertopic(bertopic_name, bertopic_topic_id):
    """Prefer parsed prefix from name; fall back to BERTopic cluster id (>=0) or -1 for outlier."""
    parsed = _canonical_topic_id_from_label(bertopic_name)
    if parsed is not None:
        return parsed
    if bertopic_topic_id == -1:
        return -1
    if bertopic_topic_id is not None and bertopic_topic_id >= 0:
        return int(bertopic_topic_id)
    return None


def _get_annotation_text(row, discussion_col, details_col):
    """Construct annotation_text from Discussion and Details per ODE spec."""
    d = str(row.get(discussion_col, "") or "").strip()
    det = str(row.get(details_col, "") or "").strip()
    if d.lower() == "nan" or d == "":
        d = ""
    if det.lower() == "nan" or det == "":
        det = ""
    if d and det:
        return d + " " + det
    if d:
        return d
    if det:
        return det
    return ""


def _corpus_statistics(docs):
    """Token stats on preprocessed whitespace-separated documents."""
    n_docs = len(docs)
    all_tokens = list(chain.from_iterable(d.split() for d in docs))
    vocab = len(set(all_tokens))
    n_tokens = len(all_tokens)
    lens = [len(d.split()) for d in docs]
    if not lens:
        return {
            "num_documents": 0,
            "vocabulary_size": 0,
            "total_tokens": 0,
            "mean_doc_length_tokens": 0.0,
            "std_doc_length_tokens": 0.0,
        }
    arr = np.asarray(lens, dtype=float)
    return {
        "num_documents": n_docs,
        "vocabulary_size": vocab,
        "total_tokens": n_tokens,
        "mean_doc_length_tokens": float(np.mean(arr)),
        "std_doc_length_tokens": float(np.std(arr, ddof=0)),
    }


def _annotation_field_nonempty(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    t = str(value).strip()
    return bool(t) and t.lower() != "nan"


def _count_codifier_empty_annotation_unclassifiable(codifier_df):
    """UNCLASSIFIABLE rows with no annotation snippet (empty Discussion+Details path in Module A)."""
    n = 0
    for _, row in codifier_df.iterrows():
        if str(row.get("llm_label", "")).strip() != "UNCLASSIFIABLE":
            continue
        if not _annotation_field_nonempty(row.get("annotation_text_used")):
            n += 1
    return n


def _documents_per_bertopic_id(doc_to_topic, n_docs):
    c = Counter(doc_to_topic.get(i, -1) for i in range(n_docs))
    return {str(int(k)): int(v) for k, v in sorted(c.items(), key=lambda x: x[0])}


def _validation_subset_bertopic_counts(validation_rows):
    c = Counter(int(r["bertopic_cluster_id"]) for r in validation_rows)
    return {str(int(k)): int(v) for k, v in sorted(c.items(), key=lambda x: x[0])}


def _codifier_row_for_csv_row_idx(codifier_df, ridx):
    """Match codifier row; tolerate float/int mismatch after read_csv."""
    m = codifier_df[codifier_df["csv_row_idx"] == ridx]
    if m.empty:
        m = codifier_df[codifier_df["csv_row_idx"] == float(ridx)]
    if m.empty:
        return None
    return m.iloc[0]


def _corpus_exclusion_from_kappa_breakdown(filenames, csv_row_to_doc, codifier_df, validation_rows):
    """
    Corpus documents that do not appear in Cohen's kappa / validation_labels (no paired expert–model row).
    """
    paper_ids = [_extract_paper_id_from_filename(f) for f in filenames]
    n_corpus = len(paper_ids)
    val_docs = {int(r["doc_index"]) for r in validation_rows}
    inv = defaultdict(list)
    for ridx, (di, _) in csv_row_to_doc.items():
        if di is not None:
            inv[int(di)].append(int(ridx))
    no_match_ids = []
    matched_empty = []
    other_excluded = []
    for doc_idx in range(n_corpus):
        if doc_idx in val_docs:
            continue
        rlist = inv.get(doc_idx, [])
        if not rlist:
            no_match_ids.append(paper_ids[doc_idx])
            continue
        ridx = rlist[0]
        row = _codifier_row_for_csv_row_idx(codifier_df, ridx)
        if row is None:
            no_match_ids.append(paper_ids[doc_idx])
            continue
        lab = str(row["llm_label"]).strip()
        ann_ok = _annotation_field_nonempty(row.get("annotation_text_used"))
        if lab == "UNCLASSIFIABLE" and not ann_ok:
            matched_empty.append({"paper_id": paper_ids[doc_idx], "csv_row_idx": ridx})
        else:
            other_excluded.append({"paper_id": paper_ids[doc_idx], "csv_row_idx": ridx, "llm_label": lab})
    return {
        "corpus_documents_not_in_kappa": int(n_corpus - len(val_docs)),
        "reason_no_csv_row_matched_to_corpus_file": len(no_match_ids),
        "reason_matched_csv_row_but_empty_expert_annotation": len(matched_empty),
        "reason_other_not_in_kappa": len(other_excluded),
        "paper_ids_no_csv_match": no_match_ids,
        "matched_but_empty_expert_annotation": matched_empty,
        "other_not_in_kappa_detail": other_excluded,
    }


def run_bertopic_with_reduce_outliers(docs, embeddings, min_topic_size=4):
    """
    Run BERTopic and apply reduce_outliers to minimize data loss from unassigned documents.
    Returns: (model, topics, probs, topic_info, topic_names_dict)
    """
    set_all_seeds()
    try:
        import hdbscan
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=1,
        )
    except Exception:
        hdbscan_model = None
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics="auto",
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )
    topics, probs = model.fit_transform(docs, embeddings=embeddings)

    # Critical: reduce outliers to address data thinning (skip if BERTopic assigned every doc)
    topics = list(topics)
    if any(t == -1 for t in topics):
        new_topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
        model.update_topics(docs, topics=new_topics)
        topics = new_topics

    topic_info = model.get_topic_info()
    topic_names_dict = {}
    for tid in topic_info["Topic"]:
        if tid == -1:
            topic_names_dict[-1] = "Outliers (unassigned)"
        else:
            info = topic_info[topic_info["Topic"] == tid]
            name = info["Name"].values[0] if len(info) and pd.notna(info["Name"].values[0]) else f"Topic {tid}"
            topic_names_dict[tid] = str(name)

    return model, topics, probs, topic_info, topic_names_dict


def _paper_topic_label(tid):
    for r in PAPER_TOPIC_LABEL_TABLE:
        if r[0] == tid:
            return r[1]
    return f"Topic {tid}"


def _umap_xy_from_bertopic(model, n_docs):
    umap_model = model.umap_model
    embeddings_hd = np.asarray(umap_model.embedding_)
    if embeddings_hd.shape[0] != n_docs:
        raise ValueError(
            f"UMAP embedding rows ({embeddings_hd.shape[0]}) != documents ({n_docs})."
        )
    umap_x = embeddings_hd[:, 0].astype(float)
    umap_y = (
        embeddings_hd[:, 1].astype(float)
        if embeddings_hd.shape[1] >= 2
        else np.zeros(n_docs, dtype=float)
    )
    return umap_x, umap_y


def _cluster_cov_ellipse(ax, xs, ys, facecolor, n_std=2.15, alpha=0.14):
    """Large semi-transparent ellipse from 2D covariance (behind scatter)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return
    mu = np.array([np.mean(xs), np.mean(ys)])
    cov = np.cov(xs, ys)
    try:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = np.maximum(vals[order], 1e-12)
        vecs = vecs[:, order]
        width = 2 * n_std * np.sqrt(vals[0])
        height = 2 * n_std * np.sqrt(vals[1])
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            facecolor=facecolor,
            edgecolor="none",
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )
        ax.add_patch(ell)
    except np.linalg.LinAlgError:
        return


def _write_umap_topic_csvs(model, topics, filenames):
    """Write umap_coordinates.csv and topic_labels.csv; print a short preview."""
    n_docs = len(topics)
    umap_x, umap_y = _umap_xy_from_bertopic(model, n_docs)
    documents = [_extract_paper_id_from_filename(f) for f in filenames]
    umap_df = pd.DataFrame({
        "document": documents,
        "umap_x": umap_x,
        "umap_y": umap_y,
        "topic": [int(t) for t in topics],
    })
    umap_df.to_csv(UMAP_COORDINATES_CSV, index=False)
    labels_df = pd.DataFrame(
        [{"topic": r[0], "label": r[1], "doc_count": r[2]} for r in PAPER_TOPIC_LABEL_TABLE]
    )
    labels_df.to_csv(TOPIC_LABELS_CSV, index=False)
    print(f"   Saved {UMAP_COORDINATES_CSV}")
    print(f"   Saved {TOPIC_LABELS_CSV}")
    print("   First 5 rows of umap_coordinates.csv:")
    print(umap_df.head(5).to_string(index=False))


def _render_umap_bertopic_ice_figure(
    model,
    topics,
    kappa,
    n_kappa_paired,
    corpus_n,
    min_topic_size=4,
):
    """
    Publication-style UMAP (BERTopic internal dims 1–2), covariance ellipses,
    and a right-hand cluster key with validation κ (subset n).
    """
    topics_arr = np.array([int(t) for t in topics], dtype=int)
    umap_x, umap_y = _umap_xy_from_bertopic(model, len(topics_arr))

    fig = plt.figure(figsize=(11.2, 7.45), facecolor="white")
    ax = fig.add_axes([0.07, 0.12, 0.56, 0.72])
    leg_ax = fig.add_axes([0.66, 0.12, 0.31, 0.72])
    leg_ax.set_facecolor("white")
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    for s in leg_ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.07,
        0.96,
        "UMAP Projection of ICE Corpus — BERTopic Topic Assignments",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        transform=fig.transFigure,
    )
    fig.text(
        0.07,
        0.925,
        "Unsupervised view: Doc2Vec embeddings → BERTopic UMAP (first two dimensions) · "
        f"min_cluster_size = {min_topic_size} · seed = 42 · N = {corpus_n} documents (full corpus).",
        fontsize=8.8,
        color="#444444",
        fontfamily="sans-serif",
        transform=fig.transFigure,
    )
    fig.text(
        0.07,
        0.895,
        "Sidebar Cohen's κ and validated n refer only to the expert–model paired subset (not N).",
        fontsize=8.2,
        color="#666666",
        fontfamily="sans-serif",
        style="italic",
        transform=fig.transFigure,
    )

    ax.set_facecolor("white")
    ax.grid(True, which="major", alpha=0.45, color="#d0d0d0", linestyle="-", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)

    unique_topics = sorted(set(topics_arr.tolist()), key=lambda x: (0 if x == -1 else 1, x))
    for tid in unique_topics:
        if tid < 0:
            continue
        m = topics_arr == tid
        if not np.any(m):
            continue
        c = BERTOPIC_UMAP_FIG_COLORS.get(tid, "#555555")
        _cluster_cov_ellipse(ax, umap_x[m], umap_y[m], facecolor=c, n_std=2.15, alpha=0.16)

    if np.any(topics_arr == -1):
        m = topics_arr == -1
        c = BERTOPIC_UMAP_FIG_COLORS[-1]
        _cluster_cov_ellipse(ax, umap_x[m], umap_y[m], facecolor=c, n_std=2.0, alpha=0.1)

    for tid in unique_topics:
        m = topics_arr == tid
        if not np.any(m):
            continue
        c = BERTOPIC_UMAP_FIG_COLORS.get(tid, "#555555")
        ax.scatter(
            umap_x[m],
            umap_y[m],
            c=c,
            s=52,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel("UMAP Dimension 2", fontsize=11, fontfamily="sans-serif")

    y = 0.98
    leg_ax.text(0.0, y, "BERTopic Clusters", fontsize=11, fontweight="bold", transform=leg_ax.transAxes)
    y -= 0.055
    leg_ax.text(
        0.0,
        y,
        f"n = {corpus_n} documents",
        fontsize=9.5,
        color="#555555",
        transform=leg_ax.transAxes,
    )
    y -= 0.09

    for tid in sorted(t for t in unique_topics if t >= 0):
        m = topics_arr == tid
        n = int(np.sum(m))
        c = BERTOPIC_UMAP_FIG_COLORS.get(tid, "#333333")
        leg_ax.text(0.0, y, f"Topic {tid}", fontsize=10, fontweight="bold", color=c, transform=leg_ax.transAxes)
        y -= 0.045
        leg_ax.text(
            0.0,
            y,
            _paper_topic_label(tid),
            fontsize=8.5,
            color="#333333",
            transform=leg_ax.transAxes,
        )
        y -= 0.065
        leg_ax.text(
            0.0,
            y,
            f"{n} documents",
            fontsize=8.5,
            color="#666666",
            transform=leg_ax.transAxes,
        )
        y -= 0.085

    if np.any(topics_arr == -1):
        n = int(np.sum(topics_arr == -1))
        leg_ax.text(
            0.0,
            y,
            f"Unassigned (−1): {n}",
            fontsize=8.5,
            color="#666666",
            transform=leg_ax.transAxes,
        )
        y -= 0.07

    for tid in unique_topics:
        if tid >= 0 and tid <= 4:
            continue
        if tid == -1:
            continue
        n = int(np.sum(topics_arr == tid))
        leg_ax.text(
            0.0,
            y,
            f"Topic id {tid}: {n} documents",
            fontsize=8.5,
            color="#333333",
            transform=leg_ax.transAxes,
        )
        y -= 0.07

    y = max(0.02, y - 0.02)
    leg_ax.plot([0, 1], [y, y], transform=leg_ax.transAxes, color="#cccccc", linewidth=0.9, clip_on=False)
    y -= 0.06
    if np.isfinite(kappa) and n_kappa_paired > 0:
        leg_ax.text(
            0.0,
            y,
            f"Cohen's κ = {kappa:.4f}",
            fontsize=10,
            fontweight="bold",
            color=BERTOPIC_UMAP_FIG_COLORS[0],
            transform=leg_ax.transAxes,
        )
        y -= 0.055
        leg_ax.text(
            0.0,
            y,
            f"{n_kappa_paired} validated documents",
            fontsize=9,
            color="#777777",
            transform=leg_ax.transAxes,
        )
    else:
        leg_ax.text(
            0.0,
            y,
            "Cohen's κ — (no paired validation rows in this run)",
            fontsize=9,
            color="#777777",
            transform=leg_ax.transAxes,
        )

    fig.savefig(UMAP_SCATTER_FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(UMAP_SCATTER_FIG_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   Saved {UMAP_SCATTER_FIG_PNG} and {UMAP_SCATTER_FIG_PDF}")


def _csv_row_by_idx(csv_df, csv_row_idx):
    """Resolve a codifier/validation csv_row_idx to a row Series."""
    if csv_row_idx is None or (isinstance(csv_row_idx, float) and np.isnan(csv_row_idx)):
        return None
    try:
        if csv_row_idx in csv_df.index:
            return csv_df.loc[csv_row_idx]
    except (KeyError, TypeError, ValueError):
        pass
    try:
        i = int(float(csv_row_idx))
    except (TypeError, ValueError):
        return None
    if 0 <= i < len(csv_df):
        return csv_df.iloc[i]
    return None


def _source_is_wikipedia_or_blog(row, source_col):
    if row is None or not source_col:
        return False
    s = str(row.get(source_col, "") or "").lower()
    if "wikipedia" in s:
        return True
    if "blog." in s or "blog/" in s or "/blog" in s:
        return True
    return False


def _discussion_nonempty_csv(row, discussion_col):
    if row is None or not discussion_col:
        return False
    d = str(row.get(discussion_col, "") or "").strip()
    return bool(d) and d.lower() != "nan"


def _canonical_int_or_none(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _validated_umap_doc_indices(validation_rows, csv_df, discussion_col, source_col):
    """
    doc_index set: κ-paired rows (both canonical ids), non-empty Discussion on
    the annotation spreadsheet row, excluding Wikipedia/blog Source URLs.
    """
    docs_ok = set()
    skipped_wiki = []
    skipped_disc = []
    for r in validation_rows:
        llm_c = _canonical_int_or_none(r.get("llm_topic_id_canonical"))
        bp_c = _canonical_int_or_none(r.get("bertopic_topic_id_canonical"))
        if llm_c is None or bp_c is None:
            continue
        di = r.get("doc_index")
        if di is None or (isinstance(di, float) and np.isnan(di)):
            continue
        di = int(di)
        crow = _csv_row_by_idx(csv_df, r.get("csv_row_idx"))
        if crow is None:
            continue
        if source_col and _source_is_wikipedia_or_blog(crow, source_col):
            skipped_wiki.append((di, str(r.get("paper_id", "")), r.get("csv_row_idx")))
            continue
        if not _discussion_nonempty_csv(crow, discussion_col):
            skipped_disc.append((di, str(r.get("paper_id", ""))))
            continue
        docs_ok.add(di)
    return docs_ok, skipped_wiki, skipped_disc


def _expert_topic_counts_on_docs(validation_rows, doc_indices):
    c = Counter()
    for r in validation_rows:
        di = r.get("doc_index")
        if di is None or (isinstance(di, float) and np.isnan(di)):
            continue
        di = int(di)
        if di not in doc_indices:
            continue
        llm_c = _canonical_int_or_none(r.get("llm_topic_id_canonical"))
        if llm_c is None:
            continue
        c[llm_c] += 1
    return c


def _topics_from_get_document_info(model, docs, topic_names_dict, doc_to_topic):
    """BERTopic per-document topics; prefers get_document_info(docs) order."""
    try:
        di = model.get_document_info(docs)
        if di is None or "Topic" not in di.columns or len(di) != len(docs):
            raise ValueError("get_document_info missing Topic or length mismatch")
        raw = pd.to_numeric(di["Topic"], errors="coerce")
        out = []
        for i in range(len(docs)):
            tid_raw = int(raw.iloc[i]) if pd.notna(raw.iloc[i]) else int(doc_to_topic.get(i, -1))
            bname = str(topic_names_dict.get(tid_raw, f"Topic {tid_raw}"))
            code = _canonical_topic_id_bertopic(bname, tid_raw)
            if code is None:
                code = tid_raw if tid_raw >= 0 else -1
            out.append(int(code))
        return out
    except Exception as e:
        print(f"   WARNING: get_document_info failed ({e}); using doc_to_topic fallback.")
        out = []
        for i in range(len(docs)):
            tid_raw = int(doc_to_topic.get(i, -1))
            bname = str(topic_names_dict.get(tid_raw, f"Topic {tid_raw}"))
            code = _canonical_topic_id_bertopic(bname, tid_raw)
            if code is None:
                code = tid_raw if tid_raw >= 0 else -1
            out.append(int(code))
        return out


def _export_validated_run_umap_bundle(
    model,
    docs,
    filenames,
    doc_to_topic,
    topic_names_dict,
    validation,
    csv_df,
    discussion_col,
    source_col,
):
    """
    Same-run UMAP coordinates + get_document_info topics; validated subset flags.
    Writes validated_run_umap_all108.csv, validated_run_umap_63.csv, and optionally validated_UMAP.png.
    """
    print("\n   Validated-run UMAP export (frozen coordinates from this fit)...")
    n_docs = len(docs)
    umap_x, umap_y = _umap_xy_from_bertopic(model, n_docs)
    documents = [_extract_paper_id_from_filename(f) for f in filenames]
    topics_code = _topics_from_get_document_info(model, docs, topic_names_dict, doc_to_topic)

    validation_rows = validation.get("validation_rows") or []
    val_docs, skipped_wiki, skipped_disc = _validated_umap_doc_indices(
        validation_rows, csv_df, discussion_col, source_col,
    )
    in_flag = [i in val_docs for i in range(n_docs)]

    df_all = pd.DataFrame({
        "document": documents,
        "umap_x": umap_x,
        "umap_y": umap_y,
        "topic": topics_code,
        "in_validated_subset": in_flag,
    })
    df_all.to_csv(VALIDATED_RUN_UMAP_ALL108, index=False)
    df_63 = df_all[df_all["in_validated_subset"]].copy()
    df_63.to_csv(VALIDATED_RUN_UMAP_63, index=False)

    print(f"   Saved {VALIDATED_RUN_UMAP_ALL108} (rows={len(df_all)})")
    print(f"   Saved {VALIDATED_RUN_UMAP_63} (rows={len(df_63)})")

    print("   Topic counts (BERTopic canonical `topic`, all 108):", dict(Counter(df_all["topic"])))
    print("   Topic counts (BERTopic canonical `topic`, validated subset):", dict(Counter(df_63["topic"])))

    expert_c = _expert_topic_counts_on_docs(validation_rows, val_docs)
    expert_dict = {k: expert_c.get(k, 0) for k in range(5)}
    print("   Expert canonical counts on validated-index set (Discussion/wiki filter):", expert_dict)

    expert_ok = expert_dict == EXPECTED_VALIDATED_EXPERT_COUNTS
    n_sub = len(df_63)
    bertopic_sub_c = Counter(df_63["topic"])
    if not all(bertopic_sub_c.get(i, 0) == EXPECTED_VALIDATED_EXPERT_COUNTS[i] for i in range(5)):
        print(
            "   NOTE: BERTopic `topic` counts from get_document_info on this subset "
            f"are {dict(bertopic_sub_c)}; paper expert table is {EXPECTED_VALIDATED_EXPERT_COUNTS}."
        )
    t4_n = int((df_63["topic"] == 4).sum())
    print(f"   Total rows in validated_run_umap_63.csv: {n_sub}")
    print(f"   Topic 4 (BERTopic canonical) point count: {t4_n}")
    print(f"   Confirmation Topic 4 has 5 points: {t4_n == 5}")

    if skipped_wiki:
        print(f"   (Excluded {len(skipped_wiki)} κ-row(s) with Wikipedia/blog Source; e.g. doc_index={skipped_wiki[0][0]})")
    if skipped_disc:
        print(f"   (Excluded {len(skipped_disc)} κ-row(s) with empty Discussion on spreadsheet)")

    kappa_docs = set()
    for r in validation_rows:
        if _canonical_int_or_none(r.get("llm_topic_id_canonical")) is None:
            continue
        if _canonical_int_or_none(r.get("bertopic_topic_id_canonical")) is None:
            continue
        di = r.get("doc_index")
        if di is None or (isinstance(di, float) and np.isnan(di)):
            continue
        kappa_docs.add(int(di))

    warn_msgs = []
    if n_sub != EXPECTED_VALIDATED_N:
        warn_msgs.append(
            f"validated_run_umap_63 has {n_sub} rows, expected {EXPECTED_VALIDATED_N}."
        )
    if not expert_ok:
        warn_msgs.append(
            f"Expert canonical counts {expert_dict} != paper expectation {EXPECTED_VALIDATED_EXPERT_COUNTS}."
        )
    if any((df_63["topic"] < 0) | (df_63["topic"] > 4)):
        warn_msgs.append("Subset contains topic id outside 0–4 (outliers or extra clusters).")

    if warn_msgs:
        print("   WARNING — validated_UMAP.png not written (CSV files still saved):")
        for w in warn_msgs:
            print(f"      - {w}")
        missing = sorted(kappa_docs - val_docs)
        extra = sorted(val_docs - kappa_docs)
        if missing:
            miss_papers = [documents[i] for i in missing if 0 <= i < len(documents)]
            print(f"      κ-eligible doc_indices excluded by Discussion/wiki filter: {missing}")
            print(f"      paper_id: {miss_papers}")
        if extra:
            print(f"      Unexpected: extra doc_indices in filtered set vs κ-only: {extra}")
        return

    try:
        _render_validated_umap_png(df_63)
        print(f"   Saved {VALIDATED_UMAP_PNG}")
    except Exception as e:
        print(f"   WARNING: validated_UMAP.png skipped ({e})")


def _render_validated_umap_png(sub_df):
    """9×6.5 in, 300 DPI; style aligned with sub_UMAP / paper figure."""
    topics_arr = sub_df["topic"].astype(int).values
    ux = sub_df["umap_x"].astype(float).values
    uy = sub_df["umap_y"].astype(float).values

    BG = "#FAFAF8"
    LEGEND_PANEL = "#F4F3EF"
    GRID = "#E5E2DA"
    SPINE = "#D0CDC6"

    labels = {r[0]: r[1] for r in PAPER_TOPIC_LABEL_TABLE}
    legend_counts = {r[0]: r[2] for r in PAPER_TOPIC_LABEL_TABLE}

    fig = plt.figure(figsize=(9, 6.5), facecolor=BG)
    ax = fig.add_axes([0.08, 0.2, 0.52, 0.68])
    leg_ax = fig.add_axes([0.63, 0.2, 0.35, 0.68])
    leg_ax.set_facecolor(LEGEND_PANEL)
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    for s in leg_ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.08,
        0.94,
        "UMAP Projection of ICE Corpus — BERTopic Topic Assignments",
        fontsize=12,
        fontweight="bold",
        family="serif",
        transform=fig.transFigure,
    )
    fig.text(
        0.08,
        0.905,
        "Doc2Vec embeddings · min_cluster_size = 4 · seed = 42",
        fontsize=9,
        color="#444444",
        family="serif",
        transform=fig.transFigure,
    )

    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=1.0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("monospace")
        lbl.set_fontsize(9)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10, family="serif")
    ax.set_ylabel("UMAP Dimension 2", fontsize=10, family="serif")

    for tid in range(5):
        m = topics_arr == tid
        if np.sum(m) >= 3:
            _cluster_cov_ellipse(ax, ux[m], uy[m], facecolor=VALIDATED_SUB_UMAP_COLORS[tid], n_std=2.1, alpha=0.08)
    if np.any(topics_arr < 0) or np.any(topics_arr > 4):
        m = (topics_arr < 0) | (topics_arr > 4)
        if np.any(m):
            _cluster_cov_ellipse(
                ax, ux[m], uy[m], facecolor=VALIDATED_SUB_UMAP_COLORS[-1], n_std=2.0, alpha=0.06,
            )

    for tid in sorted(set(topics_arr.tolist())):
        m = topics_arr == tid
        if not np.any(m):
            continue
        c = VALIDATED_SUB_UMAP_COLORS.get(tid, "#6B7280")
        ax.scatter(
            ux[m], uy[m], c=c, s=52, alpha=0.82, edgecolors="white", linewidths=0.6, zorder=2,
        )

    y = 0.99
    leg_ax.text(0.02, y, "BERTopic clusters", fontsize=10, fontweight="bold", family="serif", transform=leg_ax.transAxes)
    y -= 0.055
    leg_ax.text(
        0.02, y, f"n = {len(sub_df)} documents",
        fontsize=9, color="#444444", family="serif", transform=leg_ax.transAxes,
    )
    y -= 0.08
    for tid in range(5):
        c = VALIDATED_SUB_UMAP_COLORS[tid]
        leg_ax.text(0.02, y, f"Topic {tid}", fontsize=9.5, fontweight="bold", color=c, family="serif", transform=leg_ax.transAxes)
        y -= 0.04
        leg_ax.text(0.02, y, labels[tid], fontsize=8, color="#333333", family="serif", transform=leg_ax.transAxes)
        y -= 0.048
        leg_ax.text(
            0.02, y, f"n = {legend_counts[tid]}",
            fontsize=8.5, color="#555555", family="monospace", transform=leg_ax.transAxes,
        )
        y -= 0.072

    y -= 0.02
    leg_ax.plot([0.02, 0.98], [y, y], transform=leg_ax.transAxes, color=SPINE, linewidth=0.7)
    y -= 0.055
    leg_ax.text(
        0.02, y, "Cohen's κ = 0.7078",
        fontsize=9.5, fontweight="bold", color=VALIDATED_SUB_UMAP_COLORS[0], family="monospace", transform=leg_ax.transAxes,
    )
    y -= 0.07
    leg_ax.text(
        0.02, y, VALIDATED_UMAP_CAPTION,
        fontsize=7.5, color="#333333", family="serif", transform=leg_ax.transAxes,
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(VALIDATED_UMAP_PNG, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def _call_llm_classification(annotation_text, topic_list, topic_definitions, api_key=None):
    """Call 1: Classification. Returns exact category name from topic_list. Uses definition-based prompting."""
    formatted_codebook = "\n".join([
        f"- **{name}:** {topic_definitions.get(name, 'No definition available.')}"
        for name in topic_list
    ])
    prompt = f"""You are a domain expert in human behavior research. Your task is to classify the following expert annotation text into exactly one of the categories defined in the codebook below. First, review the codebook carefully. Then, based solely on the text, assign it to the single most appropriate category.

**Codebook:**
{formatted_codebook}

**Expert Annotation Text:**
{annotation_text[:4000]}

Respond with only the exact category name (e.g., '0_team_agent_task_norm'), and nothing else."""

    return _call_llm(prompt, api_key)


def _call_llm_confidence(annotation_text, assigned_category, api_key=None):
    """Call 2: Confidence. Returns High, Medium, or Low."""
    prompt = f"""You assigned the following expert annotation text to category "{assigned_category}". How confident are you in this classification? Respond with only one word: High, Medium, or Low.

Expert annotation text:
{annotation_text[:2000]}"""

    resp, _ = _call_llm(prompt, api_key)
    if resp:
        resp = resp.strip().capitalize()
        for level in ["High", "Medium", "Low"]:
            if level.lower() in resp.lower():
                return level
    return "Medium"


def _call_llm_rationale(annotation_text, assigned_category, api_key=None):
    """Call 3: Rationale quote. Extract verbatim phrase (max 30 words)."""
    prompt = f"""From the original expert annotation text, extract the single, verbatim phrase (maximum 30 words) that best justifies assigning it to category "{assigned_category}". Respond with only the quoted text.

Expert annotation text:
{annotation_text[:4000]}"""

    return _call_llm(prompt, api_key)[0]


def _call_llm(prompt, api_key=None):
    """Call Claude with a single pinned model. Returns (text, model_name).

    Raises RuntimeError if the API fails (no silent model switching in reported runs).
    """
    import requests

    claude_key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("Claude_API_KEY")
    if not claude_key:
        if ENABLE_GEMINI_FALLBACK:
            return _call_llm_gemini_fallback(prompt)
        raise RuntimeError(
            "CLAUDE_API_KEY is required for LLM classification. "
            "Set CLAUDE_API_KEY or run with --skip-llm / --reuse-codifier."
        )

    model_name = LLM_MODEL
    max_retries = 3
    base_delay = 30
    last_error = None

    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "max_tokens": 256,
                "temperature": LLM_TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}],
            }
            r = None
            for use_topk in (True, False):
                body = {**payload, **({"top_k": 1} if use_topk else {})}
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=body,
                    timeout=60,
                )
                if r.status_code == 400 and use_topk:
                    continue
                break
            if r.status_code == 429:
                time.sleep(base_delay * (2**attempt))
                continue
            r.raise_for_status()
            blocks = r.json().get("content", [])
            text = blocks[0].get("text", "").strip() if blocks else ""
            return text, model_name
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))

    raise RuntimeError(
        f"Claude API failed for model {model_name!r} after {max_retries} attempts: {last_error}"
    )


def _call_llm_gemini_fallback(prompt):
    """Optional Gemini path — disabled for reported validation runs (ENABLE_GEMINI_FALLBACK=False)."""
    import requests

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set and Claude unavailable.")
    max_retries = 3
    base_delay = 30
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
        for attempt in range(max_retries):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                gen_cfg = {
                    "temperature": LLM_TEMPERATURE,
                    "maxOutputTokens": 256,
                    "seed": LLM_SEED,
                }
                r = requests.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": gen_cfg,
                    },
                    params={"key": gemini_key},
                    timeout=60,
                )
                if r.status_code == 400 and "seed" in gen_cfg:
                    gen_cfg = {"temperature": LLM_TEMPERATURE, "maxOutputTokens": 256}
                    r = requests.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": gen_cfg,
                        },
                        params={"key": gemini_key},
                        timeout=60,
                    )
                if r.status_code == 429:
                    time.sleep(base_delay * (2**attempt))
                    continue
                r.raise_for_status()
                data = r.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = (parts[0].get("text", "") or "").strip() if parts else ""
                return text, model_name
            except Exception as e:
                if "404" in str(e):
                    break
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
    raise RuntimeError("Gemini fallback failed for all configured models.")


def module_a_expert_codifier(
    csv_df,
    topic_list,
    discussion_col,
    details_col,
    id_col,
    topic_definitions,
    use_llm=True,
    csv_row_to_doc=None,
):
    """
    Module A: Transform expert notes into structured labels.
    Returns DataFrame: csv_row_idx, paper_id, annotation_text_used, llm_label, llm_confidence, llm_rationale_quote

    If csv_row_to_doc is provided, LLM calls run only for rows whose Paper ID matched a document in the corpus.
    """
    rows = []
    excluded = 0
    for idx, row in csv_df.iterrows():
        paper_id = _normalize_paper_id(row.get(id_col, "")) or f"row_{idx}"
        annotation_text = _get_annotation_text(row, discussion_col, details_col)

        if csv_row_to_doc is not None:
            doc_idx, _ = csv_row_to_doc.get(idx, (None, None))
            if doc_idx is None:
                rows.append({
                    "csv_row_idx": idx,
                    "paper_id": paper_id,
                    "annotation_text_used": (annotation_text or "")[:500],
                    "llm_label": "UNCLASSIFIABLE",
                    "llm_confidence": "N/A",
                    "llm_rationale_quote": "",
                    "llm_model_used": "",
                })
                continue

        if not annotation_text or not annotation_text.strip():
            rows.append({
                "csv_row_idx": idx,
                "paper_id": paper_id,
                "annotation_text_used": "",
                "llm_label": "UNCLASSIFIABLE",
                "llm_confidence": "N/A",
                "llm_rationale_quote": "",
                "llm_model_used": "",
            })
            excluded += 1
            continue

        if not use_llm or not topic_list:
            rows.append({
                "csv_row_idx": idx,
                "paper_id": paper_id,
                "annotation_text_used": annotation_text[:500],
                "llm_label": "UNCLASSIFIABLE",
                "llm_confidence": "N/A",
                "llm_rationale_quote": "",
                "llm_model_used": "",
            })
            continue

        # Call 1: Classification (temperature=0 for reproducibility)
        llm_label, llm_model_used = _call_llm_classification(annotation_text, topic_list, topic_definitions)
        if not llm_label:
            llm_label = "UNCLASSIFIABLE"
            llm_model_used = LLM_MODEL if llm_model_used else ""
        else:
            llm_label = llm_label.strip()
            # Ensure it matches a topic name
            if llm_label not in topic_list:
                for t in topic_list:
                    if t.lower() in llm_label.lower() or llm_label.lower() in t.lower():
                        llm_label = t
                        break
                else:
                    llm_label = llm_label[:100]  # keep as-is if no match

        # Call 2: Confidence
        llm_confidence = _call_llm_confidence(annotation_text, llm_label)

        # Call 3: Rationale
        llm_rationale = _call_llm_rationale(annotation_text, llm_label)
        if llm_rationale and len(llm_rationale.split()) > 30:
            llm_rationale = " ".join(llm_rationale.split()[:30])

        rows.append({
            "csv_row_idx": idx,
            "paper_id": paper_id,
            "annotation_text_used": annotation_text[:500],
            "llm_label": llm_label,
            "llm_confidence": llm_confidence,
            "llm_rationale_quote": llm_rationale or "",
            "llm_model_used": llm_model_used or LLM_MODEL,
        })
        time.sleep(0.5)  # rate limiting

    df = pd.DataFrame(rows)
    return df, excluded


def partial_codifier_llm_for_paper_ids(
    csv_df,
    codifier_df,
    target_paper_ids,
    topic_list,
    topic_definitions,
    discussion_col,
    details_col,
    id_col,
    use_llm=True,
):
    """
    Re-run Module A LLM calls only for CSV rows whose Paper ID is in target_paper_ids.
    Updates codifier_df in place and returns it.
    """
    target_paper_ids = set(target_paper_ids)
    id_to_idx = {}
    for idx, row in csv_df.iterrows():
        pid = _normalize_paper_id(row.get(id_col, ""))
        if pid in target_paper_ids:
            id_to_idx[pid] = idx

    for pid in sorted(target_paper_ids):
        csv_idx = id_to_idx.get(pid)
        if csv_idx is None:
            print(f"   WARNING: Paper ID {pid} not found in CSV; skipping LLM reclassification.")
            continue
        row = csv_df.loc[csv_idx]
        annotation_text = _get_annotation_text(row, discussion_col, details_col)
        paper_id_out = _normalize_paper_id(row.get(id_col, "")) or f"row_{csv_idx}"

        if not annotation_text or not annotation_text.strip():
            print(f"   WARNING: No annotation text for {pid}; skipping.")
            continue

        if not use_llm or not topic_list:
            print(f"   WARNING: use_llm=False or empty topic_list; skipping {pid}.")
            continue

        print(f"   LLM reclassification: {pid} (csv_row_idx={csv_idx})")
        llm_label, llm_model_used = _call_llm_classification(annotation_text, topic_list, topic_definitions)
        if not llm_label:
            llm_label = "UNCLASSIFIABLE"
            llm_model_used = LLM_MODEL if llm_model_used else ""
        else:
            llm_label = llm_label.strip()
            if llm_label not in topic_list:
                for t in topic_list:
                    if t.lower() in llm_label.lower() or llm_label.lower() in t.lower():
                        llm_label = t
                        break
                else:
                    llm_label = llm_label[:100]

        llm_confidence = _call_llm_confidence(annotation_text, llm_label)
        llm_rationale = _call_llm_rationale(annotation_text, llm_label)
        if llm_rationale and len(llm_rationale.split()) > 30:
            llm_rationale = " ".join(llm_rationale.split()[:30])

        mask = (codifier_df["csv_row_idx"] == csv_idx) | (codifier_df["csv_row_idx"] == float(csv_idx))
        if not mask.any():
            print(f"   WARNING: No codifier row for csv_row_idx={csv_idx}; appending.")
            codifier_df = pd.concat(
                [
                    codifier_df,
                    pd.DataFrame(
                        [
                            {
                                "csv_row_idx": int(csv_idx),
                                "paper_id": paper_id_out,
                                "annotation_text_used": annotation_text[:500],
                                "llm_label": llm_label,
                                "llm_confidence": llm_confidence,
                                "llm_rationale_quote": llm_rationale or "",
                                "llm_model_used": llm_model_used or LLM_MODEL,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        else:
            pos = codifier_df.index[mask][0]
            codifier_df.loc[pos, "paper_id"] = paper_id_out
            codifier_df.loc[pos, "annotation_text_used"] = annotation_text[:500]
            codifier_df.loc[pos, "llm_label"] = llm_label
            codifier_df.loc[pos, "llm_confidence"] = llm_confidence
            codifier_df.loc[pos, "llm_rationale_quote"] = llm_rationale or ""
            codifier_df.loc[pos, "llm_model_used"] = llm_model_used or LLM_MODEL
        time.sleep(0.5)

    return codifier_df


def module_b_statistical_validation(
    codifier_df, topic_names_dict, doc_to_topic, csv_row_to_doc, csv_df,
    model, topic_info, docs,
    validation_subset_ids=None,
):
    """
    Module B: Cohen's Kappa, confidence proportions, supplementary TF-IDF.
    """
    validation_subset_ids = validation_subset_ids or load_validation_subset_ids()

    def _in_validation_subset(paper_id):
        pid = _normalize_paper_id(paper_id)
        return pid is not None and pid in validation_subset_ids

    # 1. Annotation reliability (confidence proportions) — validation subset only
    classified = codifier_df[
        (codifier_df["llm_label"] != "UNCLASSIFIABLE")
        & codifier_df["paper_id"].apply(_in_validation_subset)
    ]
    total = len(classified)
    high = (classified["llm_confidence"] == "High").sum()
    medium = (classified["llm_confidence"] == "Medium").sum()
    low = (classified["llm_confidence"] == "Low").sum()
    confidence_proportions = {
        "High": high / total if total else 0,
        "Medium": medium / total if total else 0,
        "Low": low / total if total else 0,
    }

    # 2. Cohen's Kappa: map both raters to canonical codebook ids (0..5, or -1 outlier), then compare
    rater1 = []
    rater2 = []
    rater1_topic_id = []
    rater2_topic_id = []
    validation_rows = []
    skipped_unparseable = 0
    for _, row in codifier_df.iterrows():
        if row["llm_label"] == "UNCLASSIFIABLE":
            continue
        if not _in_validation_subset(row.get("paper_id", "")):
            continue
        idx = row.get("csv_row_idx", row.name)
        doc_idx = csv_row_to_doc.get(idx, (None, None))[0] if idx in csv_row_to_doc else None
        if doc_idx is None:
            continue
        bertopic_topic_id = doc_to_topic.get(doc_idx, -1)
        bertopic_name = topic_names_dict.get(bertopic_topic_id, f"Topic {bertopic_topic_id}")
        rater1.append(row["llm_label"])
        rater2.append(bertopic_name)
        id_expert = _canonical_topic_id_from_label(row["llm_label"])
        id_model = _canonical_topic_id_bertopic(bertopic_name, bertopic_topic_id)
        if id_expert is None or id_model is None:
            skipped_unparseable += 1
            rater1_topic_id.append(None)
            rater2_topic_id.append(None)
        else:
            rater1_topic_id.append(id_expert)
            rater2_topic_id.append(id_model)
        validation_rows.append({
            "csv_row_idx": idx,
            "paper_id": row.get("paper_id", ""),
            "doc_index": doc_idx,
            "bertopic_cluster_id": bertopic_topic_id,
            "llm_label_expert": row["llm_label"],
            "bertopic_label": bertopic_name,
            "llm_topic_id_canonical": id_expert,
            "bertopic_topic_id_canonical": id_model,
            "llm_model_used": row.get("llm_model_used", LLM_MODEL),
        })

    if len(validation_rows) != len(validation_subset_ids):
        print(
            f"   WARNING: validation_rows={len(validation_rows)} but validation_subset_ids="
            f"{len(validation_subset_ids)}. Check codifier labels and corpus matching."
        )

    r1c = [a for a in rater1_topic_id if a is not None]
    r2c = [b for b in rater2_topic_id if b is not None]
    kappa = float("nan")
    if len(r1c) >= 2 and len(r1c) == len(r2c):
        if len(set(r1c)) > 1 or len(set(r2c)) > 1:
            kappa = cohen_kappa_score(r1c, r2c)
        else:
            kappa = 1.0

    # Kappa interpretation
    def interpret_kappa(k):
        if np.isnan(k):
            return "N/A"
        if k < 0:
            return "Less than chance"
        if k < 0.20:
            return "Slight"
        if k < 0.40:
            return "Fair"
        if k < 0.60:
            return "Moderate"
        if k < 0.80:
            return "Substantial"
        return "Almost Perfect"

    kappa_label = interpret_kappa(kappa)

    # 3. Supplementary: Symmetric TF-IDF (Jaccard + cosine)
    unique_topics = [t for t in np.unique(list(doc_to_topic.values())) if t != -1]
    model_docs = []
    for t in unique_topics:
        mask = [doc_to_topic.get(i, -1) == t for i in range(len(docs))]
        topic_docs = " ".join(docs[i] for i in range(len(docs)) if mask[i])
        model_docs.append(topic_docs)

    # annotation_texts from codifier (exclude UNCLASSIFIABLE)
    human_texts = []
    for idx, row in codifier_df.iterrows():
        if row["llm_label"] != "UNCLASSIFIABLE" and str(row.get("annotation_text_used", "")).strip():
            human_texts.append(str(row["annotation_text_used"]).strip())

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1)
    all_docs = model_docs + human_texts
    if all_docs:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        n_model = len(model_docs)
        n_human = len(human_texts)
        model_vectors = tfidf_matrix[:n_model]
        human_vectors = tfidf_matrix[n_model:] if n_human > 0 else None
        if human_vectors is not None and human_vectors.shape[0] > 0:
            sim_matrix = cosine_similarity(model_vectors, human_vectors)
            mean_cosine = float(np.mean(sim_matrix))
        else:
            mean_cosine = 0.0

        feature_names = vectorizer.get_feature_names_out()
        top_n = 30
        all_model_words = set()
        for i in range(n_model):
            scores = tfidf_matrix[i].toarray().flatten()
            top_idx = np.argsort(scores)[::-1][:top_n]
            for j in top_idx:
                if scores[j] > 0:
                    all_model_words.add(feature_names[j])
        all_human_words = set()
        for i in range(n_human):
            scores = tfidf_matrix[n_model + i].toarray().flatten()
            top_idx = np.argsort(scores)[::-1][:top_n]
            for j in top_idx:
                if scores[j] > 0:
                    all_human_words.add(feature_names[j])
        overlap = all_model_words & all_human_words
        union = all_model_words | all_human_words
        jaccard = len(overlap) / len(union) if union else 0
    else:
        mean_cosine = 0.0
        jaccard = 0.0

    return {
        "confidence_proportions": confidence_proportions,
        "kappa": kappa,
        "kappa_interpretation": kappa_label,
        "rater1_labels": rater1,
        "rater2_labels": rater2,
        "rater1_topic_id": r1c,
        "rater2_topic_id": r2c,
        "skipped_unparseable_topic_id": skipped_unparseable,
        "validation_rows": validation_rows,
        "jaccard_supplementary": jaccard,
        "mean_cosine_supplementary": mean_cosine,
    }


def module_c_latent_variables(model, topic_info, docs, doc_to_topic, topic_names_dict):
    """
    Module C: 2-3 latent variables per BERTopic theme with grounding and operationalization.
    """
    results = []
    unique_topics = [t for t in sorted(topic_info["Topic"].unique()) if t != -1]
    for tid in unique_topics:
        topic_words = model.get_topic(tid)
        if not topic_words:
            continue
        top_terms = [w for w, _ in topic_words[:15]]
        topic_docs = [docs[i] for i in range(len(docs)) if doc_to_topic.get(i, -1) == tid]
        topic_name = topic_names_dict.get(tid, f"Topic {tid}")

        # Use LLM to suggest 2-3 latent variables (optional - can be rule-based)
        variables = []
        corpus_snippet = " ".join(top_terms) + " " + " ".join(topic_docs[:3])[:2000]
        prompt = f"""For the BERTopic theme "{topic_name}" with top terms: {', '.join(top_terms[:10])}, identify 2-3 specific latent variables suitable for agent-based modeling. For each variable provide:
1. Variable name (e.g., "conflict frequency per mission week")
2. Grounding term or phrase from the corpus
3. Suggested operationalization for simulation (e.g., "modeled as Poisson process with rate λ")

Corpus excerpt:
{corpus_snippet[:3000]}

Respond in this exact format for each variable:
- NAME: [variable name]
- GROUNDING: [phrase from corpus]
- OPERATIONALIZATION: [simulation suggestion]"""

        resp = _call_llm(prompt)
        if resp:
            for block in resp.split("- NAME:"):
                if "GROUNDING:" in block and "OPERATIONALIZATION:" in block:
                    lines = block.strip().split("\n")
                    name = lines[0].strip().strip("-").strip()
                    grounding = ""
                    oper = ""
                    for L in lines[1:]:
                        if L.strip().startswith("GROUNDING:"):
                            grounding = L.replace("GROUNDING:", "").strip()
                        elif L.strip().startswith("OPERATIONALIZATION:"):
                            oper = L.replace("OPERATIONALIZATION:", "").strip()
                    if name:
                        variables.append({"name": name, "grounding": grounding, "operationalization": oper})
        if len(variables) < 2:
            # Fallback: create from top terms
            for term in top_terms[:3]:
                if term and term not in [v["name"] for v in variables]:
                    variables.append({
                        "name": f"{term.replace('_', ' ')} (from c-TF-IDF)",
                        "grounding": term,
                        "operationalization": f"Modeled as a Poisson process with rate λ estimated from document frequency of '{term}'.",
                    })
                    if len(variables) >= 3:
                        break

        results.append({"topic_id": tid, "topic_name": topic_name, "variables": variables[:3]})
    return results


def create_confusion_matrix_plot(rater1, rater2, output_path):
    """Normalized confusion matrix: Rows=BERTopic (model), Cols=LLM (expert). Uses canonical topic ids."""
    if not rater1 or not rater2:
        return
    labels_r = sorted(set(rater2))
    labels_c = sorted(set(rater1))
    n_r, n_c = len(labels_r), len(labels_c)
    cm = np.zeros((n_r, n_c))
    for i in range(len(rater2)):
        r, c = rater2[i], rater1[i]
        if r in labels_r and c in labels_c:
            ir, ic = labels_r.index(r), labels_c.index(c)
            cm[ir, ic] += 1
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, n_c * 1.2), max(8, n_r * 0.8)))
    xticklabels = [f"id {l}" for l in labels_c]
    yticklabels = [f"id {l}" for l in labels_r]
    sns.heatmap(
        cm_norm,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Normalized proportion"},
    )
    ax.set_xlabel("LLM-Codified Expert Consensus Labels", fontsize=12)
    ax.set_ylabel("BERTopic Model Labels", fontsize=12)
    ax.set_title("Normalized Confusion Matrix: BERTopic vs LLM Expert Consensus", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def _save_bertopic_state(model, d2v_model, topic_names_dict, filenames, topics):
    """Persist fitted BERTopic + Doc2Vec + per-paper topic ids for --validation-only runs."""
    paper_ids = [_extract_paper_id_from_filename(f) for f in filenames]
    paper_id_to_topic = {paper_ids[i]: int(topics[i]) for i in range(len(topics))}
    state = {
        "bertopic": model,
        "doc2vec": d2v_model,
        "topic_names_dict": topic_names_dict,
        "paper_id_to_topic": paper_id_to_topic,
    }
    try:
        with open(BERTOPIC_STATE_PATH, "wb") as f:
            pickle.dump(state, f)
        print(f"   Saved BERTopic validation state to {BERTOPIC_STATE_PATH}")
    except Exception as e:
        print(f"   WARNING: Could not save BERTopic state ({e}). --validation-only will not be available until a full run succeeds.")


def _finalize_pipeline(
    args,
    csv_df,
    discussion_col,
    details_col,
    id_col,
    topic_definitions,
    model,
    topic_info,
    topic_names_dict,
    doc_to_topic,
    csv_row_to_doc,
    docs,
    filenames,
    validation_only=False,
    corpus_stats=None,
):
    """Module A (optional), B, C, reports — shared by full and validation-only runs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    topic_names_by_id = [topic_names_dict[t] for t in sorted(topic_names_dict.keys()) if t != -1]
    # Alphabetical codebook order in prompts for deterministic LLM inputs across runs
    topic_list = sorted(topic_names_by_id)
    validation_subset_ids = load_validation_subset_ids()
    print(
        f"   Validation subset: n={len(validation_subset_ids)} Paper IDs from "
        f"{os.path.relpath(VALIDATION_SUBSET_IDS_PATH, PROJECT_ROOT)}"
    )

    codifier_path = os.path.join(RESULTS_DIR, "module_a_codifier_output.csv")
    if getattr(args, "extend_validation", False):
        print("\n2a. Extend validation — LLM reclassification for newly matched papers only...")
        if not os.path.exists(codifier_path):
            print(f"ERROR: {codifier_path} required for --extend-validation.")
            sys.exit(1)
        codifier_df_patch = pd.read_csv(codifier_path)
        codifier_df_patch = partial_codifier_llm_for_paper_ids(
            csv_df,
            codifier_df_patch,
            EXTEND_FORCE_INFER_PAPER_IDS,
            topic_list,
            topic_definitions,
            discussion_col,
            details_col,
            id_col,
            use_llm=not args.skip_llm,
        )
        codifier_df_patch.to_csv(codifier_path, index=False)
        print(f"   Updated codifier rows for: {', '.join(sorted(EXTEND_FORCE_INFER_PAPER_IDS))}")
        args.reuse_codifier = True

    # Module A
    print("\n2. Module A — Expert Consensus Codifier...")
    if args.reuse_codifier:
        if not os.path.exists(codifier_path):
            print(f"ERROR: --reuse-codifier requires existing file: {codifier_path}")
            sys.exit(1)
        codifier_df = pd.read_csv(codifier_path)
        if "llm_model_used" not in codifier_df.columns:
            codifier_df["llm_model_used"] = ""
        excluded_count = _count_codifier_empty_annotation_unclassifiable(codifier_df)
        print(f"   Loaded existing codifier from {codifier_path}")
    else:
        codifier_df, excluded_count = module_a_expert_codifier(
            csv_df,
            topic_list,
            discussion_col,
            details_col,
            id_col,
            topic_definitions,
            use_llm=not args.skip_llm,
            csv_row_to_doc=csv_row_to_doc,
        )
        codifier_df.to_csv(codifier_path, index=False)
    print(f"   Codifier rows UNCLASSIFIABLE with empty annotation text: {excluded_count}")
    n_matched = sum(
        1 for idx in csv_df.index
        if csv_row_to_doc.get(idx, (None, None))[0] is not None
    )
    print(f"   CSV rows with matched document in corpus: {n_matched}")

    # Module B
    print("\n3. Module B — Statistical Validation...")
    validation = module_b_statistical_validation(
        codifier_df, topic_names_dict, doc_to_topic, csv_row_to_doc,
        csv_df, model, topic_info, docs,
        validation_subset_ids=validation_subset_ids,
    )
    print(f"   Confidence: High={validation['confidence_proportions']['High']:.2%}, "
          f"Medium={validation['confidence_proportions']['Medium']:.2%}, Low={validation['confidence_proportions']['Low']:.2%}")
    print(f"   Cohen's Kappa (canonical topic id 0–5 / -1): {validation['kappa']:.4f} ({validation['kappa_interpretation']})")
    if validation.get("skipped_unparseable_topic_id", 0):
        print(f"   Skipped rows (unparseable topic id): {validation['skipped_unparseable_topic_id']}")
    print(f"   Supplementary Jaccard: {validation['jaccard_supplementary']:.4f}, "
          f"Cosine: {validation['mean_cosine_supplementary']:.4f}")

    val_labels_path = os.path.join(RESULTS_DIR, "validation_labels.csv")
    pd.DataFrame(validation["validation_rows"]).to_csv(val_labels_path, index=False)
    print(f"   Saved {val_labels_path}")

    r1c = validation["rater1_topic_id"]
    r2c = validation["rater2_topic_id"]
    n_kappa = len(r1c)
    n_corpus = len(docs)
    label_ids = sorted(set(r1c) | set(r2c))
    id_names = [f"topic_id_{i}" for i in label_ids]
    report_lines = [
        "Labels are canonical codebook ids: integer prefix before first '_' in topic names (e.g. 0_team_agent_task_norm -> 0).",
        "Outliers / unassigned -> -1. Cohen's kappa and this report use these ids for both expert (LLM) and BERTopic.",
        "Classification report: y_true = expert topic id, y_pred = BERTopic topic id (same coding).",
        "Per-class support = count of that expert id in y_true.",
        f"SCOPE: This report and figures/confusion_matrix.png describe ONLY the validation subset (n = {n_kappa} paired documents),",
        f"not all corpus files (N = {n_corpus}). Do not report these metrics as corpus-wide.",
        "",
    ]
    if len(r1c) == 0:
        report_lines.append("No paired canonical labels; skipped classification_report.")
    else:
        report_lines.append(
            classification_report(
                r1c,
                r2c,
                labels=label_ids,
                target_names=id_names,
                zero_division=0,
            )
        )
    class_report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(class_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"   Saved {class_report_path}")

    latent_results = []
    if not args.skip_latent:
        print("\n4. Module C — Latent Variable Prototypes...")
        latent_results = module_c_latent_variables(
            model, topic_info, docs, doc_to_topic, topic_names_dict
        )

    create_confusion_matrix_plot(
        validation["rater1_topic_id"],
        validation["rater2_topic_id"],
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
    )

    topics_row_order = [int(doc_to_topic.get(i, -1)) for i in range(len(docs))]
    try:
        _render_umap_bertopic_ice_figure(
            model,
            topics_row_order,
            float(validation["kappa"]),
            n_kappa,
            n_corpus,
            min_topic_size=4,
        )
    except Exception as e:
        print(f"   WARNING: UMAP publication figure skipped ({e})")

    try:
        source_col = next((c for c in csv_df.columns if str(c).strip().lower() == "source"), None)
        _export_validated_run_umap_bundle(
            model,
            docs,
            filenames,
            doc_to_topic,
            topic_names_dict,
            validation,
            csv_df,
            discussion_col,
            source_col,
        )
    except Exception as e:
        print(f"   WARNING: validated-run UMAP export skipped ({e})")

    paper_ids_out = [_extract_paper_id_from_filename(f) for f in filenames]
    val_doc_idx = {int(r["doc_index"]) for r in validation["validation_rows"]}
    assign_rows = []
    for i in range(len(docs)):
        tid = int(doc_to_topic.get(i, -1))
        assign_rows.append({
            "doc_index": i,
            "paper_id": paper_ids_out[i],
            "filename": os.path.basename(filenames[i]),
            "bertopic_topic_id": tid,
            "bertopic_label": topic_names_dict.get(tid, f"Topic {tid}"),
            "in_validation_subset": i in val_doc_idx,
        })
    assign_path = os.path.join(TOPIC_ASSIGNMENTS_DIR, "document_topic_assignments.csv")
    pd.DataFrame(assign_rows).to_csv(assign_path, index=False)
    print(f"   Saved {assign_path}")

    kappa_exclusion = _corpus_exclusion_from_kappa_breakdown(
        filenames, csv_row_to_doc, codifier_df, validation["validation_rows"],
    )
    docs_per_topic = _documents_per_bertopic_id(doc_to_topic, len(docs))
    val_subset_topic_counts = _validation_subset_bertopic_counts(validation["validation_rows"])
    n_unclass_total = int((codifier_df["llm_label"] == "UNCLASSIFIABLE").sum())

    n_non_out = int((topic_info["Topic"] != -1).sum()) if topic_info is not None and len(topic_info) else 0
    methodology_summary = {
        "pipeline_version": "3.0",
        "timestamp": datetime.now().isoformat(),
        "seeds": {
            "REPRODUCIBILITY_SEED": REPRODUCIBILITY_SEED,
            "DOC2VEC_SEED": DOC2VEC_SEED,
            "LLM_SEED": LLM_SEED,
        },
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_gemini_fallback_enabled": ENABLE_GEMINI_FALLBACK,
        "python_version": sys.version.split()[0],
        "validation_subset_ids_file": os.path.relpath(VALIDATION_SUBSET_IDS_PATH, PROJECT_ROOT),
        "validation_subset_n": len(validation_subset_ids),
        "topic_list_used_at_runtime": topic_names_by_id,
        "topic_list_prompt_order_alphabetical": topic_list,
        "bertopic_num_topics_non_outlier": n_non_out,
        "corpus_statistics": corpus_stats or _corpus_statistics(docs),
        "topic_definitions": topic_definitions,
        "prompts": {
            "classification": "Definition-based prompting. Template: 'You are a domain expert in human behavior research. Your task is to classify the following expert annotation text into exactly one of the categories defined in the codebook below. First, review the codebook carefully. Then, based solely on the text, assign it to the single most appropriate category. **Codebook:** {formatted_codebook} **Expert Annotation Text:** {annotation_text} Respond with only the exact category name (e.g., 0_team_agent_task_norm), and nothing else.'",
            "confidence": "How confident are you in this classification? Respond with only one word: High, Medium, or Low.",
            "rationale": "From the original expert annotation text, extract the single, verbatim phrase (maximum 30 words) that best justifies your classification. Respond with only the quoted text.",
        },
        "documents_processed": len(docs),
        "documents_classified": len(codifier_df[codifier_df["llm_label"] != "UNCLASSIFIABLE"]),
        "documents_excluded_empty_expert_annotation": excluded_count,
        "documents_excluded": excluded_count,
        "codifier_rows_unclassifiable_total": n_unclass_total,
        "documents_per_bertopic_id_full_corpus": docs_per_topic,
        "documents_per_bertopic_id_validation_subset_only": val_subset_topic_counts,
        "corpus_exclusion_from_cohens_kappa": kappa_exclusion,
        "irr_note": "Fleiss' Kappa was not used because the human annotation is a collaborative consensus, not independent parallel coding.",
        "cohen_kappa_uses_canonical_topic_ids": True,
        "skipped_unparseable_topic_id": validation.get("skipped_unparseable_topic_id", 0),
        "validation_only_mode": validation_only,
        "extend_validation": getattr(args, "extend_validation", False),
        "bertopic_state_path": getattr(args, "resolved_bertopic_state_path", None),
        "kappa_paired_rows": len(validation["rater1_topic_id"]),
        "cohen_kappa": float(validation["kappa"]) if not np.isnan(validation["kappa"]) else None,
        "paper_reporting": {
            "corpus_n_full_bertopic": len(docs),
            "validation_n_cohen_kappa_classification_report_confusion_matrix": n_kappa,
            "documents_excluded_field_definition": (
                "documents_excluded_empty_expert_annotation: CSV codifier rows with UNCLASSIFIABLE and no annotation text "
                "(empty Discussion+Details). This is NOT the count of corpus files excluded from kappa; see corpus_exclusion_from_cohens_kappa."
            ),
            "full_corpus_topic_counts_file": "document_topic_assignments.csv",
            "validation_pairwise_file": "validation_labels.csv",
            "classification_report_and_confusion_matrix_scope": (
                "Precision, recall, classification_report.txt, and results/figures/confusion_matrix.png apply only to the n validation "
                "pairs in validation_labels.csv, not to all N corpus documents."
            ),
            "ablation_study_scope": (
                "Embedding × min_topic_size ablation is implemented in run_pipeline.py (TF-IDF, Doc2Vec, SciBERT, SocialBERT; "
                "min_topic_size in {2,4,6,8,10}). It loads whatever texts are in data/corpus at run time. "
                "This repository does not ship a regenerated ablation table for the expanded corpus under results/; "
                "older ablation_study_results.csv files in parent 'Old Ver Work folder' reflect the earlier smaller corpus. "
                "Re-run run_pipeline.py after changing processed_data to refresh ablation numbers for the current N."
            ),
        },
    }
    with open(os.path.join(RESULTS_DIR, "methodology_summary.json"), "w") as f:
        json.dump(methodology_summary, f, indent=2)

    print(
        f"   Corpus ↔ κ scope: N={len(docs)} BERTopic documents; "
        f"n={n_kappa} paired for κ / classification_report / confusion_matrix; "
        f"{kappa_exclusion['corpus_documents_not_in_kappa']} corpus files not in κ pairing "
        f"(no CSV match: {kappa_exclusion['reason_no_csv_row_matched_to_corpus_file']}, "
        f"empty expert text: {kappa_exclusion['reason_matched_csv_row_but_empty_expert_annotation']}). "
        f"See methodology_summary.json → paper_reporting and document_topic_assignments.csv."
    )

    outlier_llm_labels = []
    for _, row in codifier_df.iterrows():
        idx = row.get("csv_row_idx", row.name)
        doc_idx = csv_row_to_doc.get(idx, (None, None))[0] if idx in csv_row_to_doc else None
        if doc_idx is not None and doc_to_topic.get(doc_idx, -1) == -1:
            if row["llm_label"] != "UNCLASSIFIABLE":
                outlier_llm_labels.append(row["llm_label"])
    outlier_themes = pd.Series(outlier_llm_labels).value_counts().to_dict() if outlier_llm_labels else {}

    report = f"""# Blue Ocean Pipeline v3.0 — Master Report

## 1. Annotation Reliability

Proportion of documents assigned **High**, **Medium**, and **Low** confidence by the LLM:
- **High**: {validation['confidence_proportions']['High']:.2%}
- **Medium**: {validation['confidence_proportions']['Medium']:.2%}
- **Low**: {validation['confidence_proportions']['Low']:.2%}

*Interpretation*: The distribution of confidence levels serves as a proxy for the ambiguity of the expert notes. A higher proportion of High confidence indicates clearer, more consistent expert annotations.

## 2. Primary Validation (Cohen's Kappa)

- **Corpus size (BERTopic)**: {len(docs)} documents (full `data/corpus` run).
- **Kappa / classification-report / confusion-matrix subset**: **n = {n_kappa}** paired documents with matched CSV row, non-empty expert text, and LLM label not UNCLASSIFIABLE (see `validation_labels.csv`). These metrics are **not** computed over all {len(docs)} files.
- **Kappa score**: {validation['kappa']:.4f} (computed on **canonical topic ids**: leading integer in each topic name, e.g. `0_...` → 0; outliers → -1)
- **Interpretation**: {validation['kappa_interpretation']}

*Interpretation*: Cohen's Kappa quantifies agreement between the LLM-codified expert consensus and the BERTopic model's topic assignments after mapping both string labels to the same numeric codebook id. The {validation['kappa_interpretation']} level of agreement suggests that the BERTopic pipeline captures themes that align with expert judgment when used in conjunction with expert validation.

## 3. Supplementary Lexical Check (Symmetric TF-IDF)

*Label: SUPPLEMENTARY — word-level overlap, not primary validation.*

- **Jaccard similarity**: {validation['jaccard_supplementary']:.4f}
- **Mean cosine similarity**: {validation['mean_cosine_supplementary']:.4f}

## 4. Confusion Matrix

Row-normalized heatmap for the **same n = {n_kappa} validation pairs** as Section 2 (not the full corpus).

![Confusion Matrix](figures/confusion_matrix.png)

## 5. Latent Theme Analysis (from former outliers)

Documents originally classified by BERTopic as outliers (Topic -1) were re-examined by the LLM. Emergent themes assigned by the LLM:

{json.dumps(outlier_themes, indent=2) if outlier_themes else "No outlier documents were assigned non-UNCLASSIFIABLE labels by the LLM."}

*Interpretation*: These themes represent patterns the density-based clustering missed but that expert-informed LLM classification identified.

## 6. Latent Variable Prototypes

"""
    for item in latent_results:
        report += f"\n### {item['topic_name']}\n\n"
        for v in item["variables"]:
            report += f"- **{v['name']}**\n"
            report += f"  - Grounding: {v['grounding']}\n"
            report += f"  - Operationalization: {v['operationalization']}\n\n"

    report += f"""
## 7. Pipeline Architecture Description (for Visualization)

```mermaid
flowchart TB
    subgraph Sources
        A["{len(docs)} documents — full text corpus"]
        B[Expert Annotation CSV - Discussion + Details]
    end

    subgraph BERTopic
        A --> C[Doc2Vec Embeddings]
        C --> D[UMAP + HDBSCAN]
        D --> E[reduce_outliers]
        E --> F[Topic Assignments]
    end

    subgraph ExpertCodifier
        B --> G[annotation_text]
        G --> H[LLM Classification]
        H --> I[llm_label, confidence, rationale]
    end

    subgraph Validation
        F --> J[Cohen's Kappa]
        I --> J
        F --> K[Supplementary TF-IDF]
        B --> K
    end

    J --> L[Master Report]
    K --> L
```

*Inspired by OASIS paper's convergent two-source independent validation diagram.*
"""

    with open(os.path.join(RESULTS_DIR, "master_report.md"), "w") as f:
        f.write(report)
    print(f"\nMaster report saved to {os.path.join(RESULTS_DIR, 'master_report.md')}")

    print("\n" + "=" * 70)
    print("Blue Ocean Pipeline complete. Outputs in:", RESULTS_DIR)
    print("=" * 70)


def run_validation_only_mode(args):
    """
    Recompute Module B (and downstream outputs) using saved BERTopic/Doc2Vec state.
    Assignments for known paper IDs are taken from the state; new files get infer_vector + transform only.
    With --extend-validation, loads frozen 78-doc state and forces infer+transform for the four extended papers.
    """
    if getattr(args, "extend_validation", False):
        state_path = args.bertopic_state or BERTOPIC_STATE_78DOC_PATH
    else:
        state_path = args.bertopic_state or BERTOPIC_STATE_PATH

    if not os.path.exists(state_path):
        print(f"ERROR: BERTopic state not found at {state_path}")
        if getattr(args, "extend_validation", False):
            print("Create it by running a full pipeline on the 78-document corpus (without Palinkas), then:")
            print(f"  cp {BERTOPIC_STATE_PATH} {BERTOPIC_STATE_78DOC_PATH}")
        else:
            print("Run a full pipeline once (without --validation-only) to fit BERTopic and save state.")
        sys.exit(1)

    set_all_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TOPIC_ASSIGNMENTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    if getattr(args, "extend_validation", False):
        print("Blue Ocean Pipeline — extend validation (frozen 78-doc model + 4 infer/transform)")
    else:
        print("Blue Ocean Pipeline — validation-only (no BERTopic refit)")
    print(f"State file: {state_path}")
    print("=" * 70)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)
    csv_df = pd.read_csv(CSV_PATH)
    discussion_col = [c for c in csv_df.columns if "Discussion" in c][0] if any("Discussion" in c for c in csv_df.columns) else csv_df.columns[7]
    details_col = [c for c in csv_df.columns if "Details" in c][0] if any("Details" in c for c in csv_df.columns) else csv_df.columns[8]
    id_col = [c for c in csv_df.columns if "Paper ID" in c][0] if any("Paper ID" in c for c in csv_df.columns) else csv_df.columns[0]

    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Processed data not found at {INPUT_FOLDER}")
        sys.exit(1)
    raw_docs, filenames = load_data_from_drive(INPUT_FOLDER)
    custom_stopwords = get_custom_stopwords()
    docs = preprocess_text(raw_docs, min_word_length=3, custom_stopwords=custom_stopwords)
    print(f"Loaded {len(docs)} documents")

    csv_row_to_doc = _match_csv_row_to_doc(csv_df, filenames)

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    model = state["bertopic"]
    d2v_model = state["doc2vec"]
    topic_names_dict = state["topic_names_dict"]
    saved_map = state["paper_id_to_topic"]

    force_infer = EXTEND_FORCE_INFER_PAPER_IDS if getattr(args, "extend_validation", False) else frozenset()

    paper_ids = [_extract_paper_id_from_filename(f) for f in filenames]
    doc_to_topic = {}
    for i, pid in enumerate(paper_ids):
        use_infer = (pid not in saved_map) or (pid in force_infer)
        if not use_infer:
            doc_to_topic[i] = saved_map[pid]
        else:
            emb = d2v_model.infer_vector(docs[i].split())
            new_topics, _ = model.transform([docs[i]], embeddings=np.array([emb]))
            doc_to_topic[i] = int(new_topics[0])
            if pid in force_infer:
                print(f"   infer+transform (extend): {pid} -> BERTopic topic {doc_to_topic[i]}")
            else:
                print(f"   New corpus file (not in saved state): {pid} -> BERTopic topic {doc_to_topic[i]}")

    topic_info = model.get_topic_info()
    topic_definitions = load_topic_definitions(CODEBOOK_CSV)

    args.resolved_bertopic_state_path = state_path

    _finalize_pipeline(
        args, csv_df, discussion_col, details_col, id_col, topic_definitions,
        model, topic_info, topic_names_dict, doc_to_topic, csv_row_to_doc, docs,
        filenames,
        validation_only=True,
        corpus_stats=_corpus_statistics(docs),
    )


def main():
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls (use mock labels for testing)")
    parser.add_argument("--skip-latent", action="store_true", help="Skip Module C latent variable extraction")
    parser.add_argument(
        "--reuse-codifier",
        action="store_true",
        help="Load Module A output from results/module_a_codifier_output.csv instead of calling the LLM",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Skip BERTopic refit; reuse saved state from results/bertopic_validation_state.pkl (run full pipeline once first).",
    )
    parser.add_argument(
        "--extend-validation",
        action="store_true",
        help="With --validation-only: use frozen 78-doc state, infer+transform for four extended papers, partial LLM codifier.",
    )
    parser.add_argument(
        "--bertopic-state",
        default=None,
        metavar="PATH",
        help="Pickle from _save_bertopic_state (default: bertopic_validation_state.pkl, or bertopic_validation_state_78doc.pkl with --extend-validation).",
    )
    parser.add_argument(
        "--embedding",
        default="doc2vec",
        choices=("doc2vec", "scibert", "socialbert", "tfidf"),
        help="Document embeddings before BERTopic (default: doc2vec; matches ablation study).",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=4,
        metavar="N",
        help="BERTopic min_topic_size / HDBSCAN min_cluster_size (default: 4).",
    )
    args = parser.parse_args()

    if args.extend_validation and not args.validation_only:
        print("ERROR: --extend-validation requires --validation-only.")
        sys.exit(1)

    if args.validation_only:
        run_validation_only_mode(args)
        return

    set_all_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TOPIC_ASSIGNMENTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("Blue Ocean Pipeline v3.0 — ODE-Compliant Analysis")
    print("=" * 70)

    # Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)
    csv_df = pd.read_csv(CSV_PATH)
    discussion_col = [c for c in csv_df.columns if "Discussion" in c][0] if any("Discussion" in c for c in csv_df.columns) else csv_df.columns[7]
    details_col = [c for c in csv_df.columns if "Details" in c][0] if any("Details" in c for c in csv_df.columns) else csv_df.columns[8]
    id_col = [c for c in csv_df.columns if "Paper ID" in c][0] if any("Paper ID" in c for c in csv_df.columns) else csv_df.columns[0]

    # Load documents
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Processed data not found at {INPUT_FOLDER}")
        sys.exit(1)
    raw_docs, filenames = load_data_from_drive(INPUT_FOLDER)
    custom_stopwords = get_custom_stopwords()
    docs = preprocess_text(raw_docs, min_word_length=3, custom_stopwords=custom_stopwords)
    print(f"Loaded {len(docs)} documents")

    # Build mappings
    csv_row_to_doc = _match_csv_row_to_doc(csv_df, filenames)

    # BERTopic with reduce_outliers
    print("\n1. Running BERTopic with reduce_outliers=True...")
    paper_ids = [_extract_paper_id_from_filename(f) for f in filenames]
    d2v_model = None
    if args.embedding == "doc2vec":
        embeddings, d2v_model = get_doc2vec_embedding(
            docs, seed=DOC2VEC_SEED, workers=1, doc_keys=paper_ids
        )
    elif args.embedding == "scibert":
        embeddings, _ = get_bert_embedding(docs, model_name="allenai/scibert_scivocab_uncased")
    elif args.embedding == "socialbert":
        embeddings, _ = get_bert_embedding(docs, model_name="ESGBERT/SocialBERT-social")
    else:
        embeddings, _ = get_tfidf_embedding(docs)
    if embeddings is None:
        print(f"ERROR: Failed to build embeddings for --embedding {args.embedding}")
        sys.exit(1)

    print(
        f"   Using embedding={args.embedding}, min_topic_size={args.min_topic_size} "
        f"(Doc2Vec workers=1 when doc2vec; all seeds via set_all_seeds)"
    )
    model, topics, probs, topic_info, topic_names_dict = run_bertopic_with_reduce_outliers(
        docs, embeddings, min_topic_size=args.min_topic_size
    )
    doc_to_topic = {i: t for i, t in enumerate(topics)}
    topic_list = [topic_names_dict[t] for t in sorted(topic_names_dict.keys()) if t != -1]
    print(f"   Topics: {topic_list}")

    n_non_outlier_topics = int((topic_info["Topic"] != -1).sum()) if topic_info is not None else 0
    if n_non_outlier_topics == 5:
        _save_bertopic_state(model, d2v_model, topic_names_dict, filenames, topics)
        print("5 TOPICS CONFIRMED — STATE SAVED")
    else:
        print(
            f"STOP: Non-outlier BERTopic topics = {n_non_outlier_topics} (expected 5). "
            f"Configuration: embedding={args.embedding}, min_topic_size={args.min_topic_size}."
        )
        print("   Not running Module A or saving bertopic_validation_state.pkl.")
        sys.exit(2)

    _write_umap_topic_csvs(model, topics, filenames)

    topic_definitions = build_runtime_topic_definitions(
        model,
        topic_names_dict,
        os.path.join(CODEBOOK_CSV),
    )

    _finalize_pipeline(
        args, csv_df, discussion_col, details_col, id_col, topic_definitions,
        model, topic_info, topic_names_dict, doc_to_topic, csv_row_to_doc, docs,
        filenames,
        validation_only=False,
        corpus_stats=_corpus_statistics(docs),
    )


if __name__ == "__main__":
    main()
