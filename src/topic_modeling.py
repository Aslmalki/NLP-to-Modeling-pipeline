"""
Topic modeling module for NLP Pipeline.
ODE-compliant: UMAP random_state=42 for reproducibility.
"""
import os
from collections import defaultdict

import numpy as np
from config import TOPIC_LABELS
import pandas as pd
from bertopic import BERTopic
from gensim import corpora
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
import umap


def optimize_bertopic_ablation(docs, embeddings_dict, min_size_range, tokenized_docs=None):
    """
    Perform topic size and n-gram optimization with ablation studies.

    Args:
        docs: List of documents.
        embeddings_dict: Dictionary of embeddings (e.g., {'TFIDF': embeddings_tfidf, ...}).
        min_size_range: Range of minimum topic sizes to evaluate.
        tokenized_docs: Optional pre-tokenized docs for coherence.

    Returns:
        dict: Results dictionary containing Silhouette scores, Coherence scores, topic counts, etc.

    Note:
        Top2Vec document_vectors are used as embeddings only. All configurations use
        BERTopic UMAP and HDBSCAN for clustering so results are directly comparable
        across all five embedding methods.
    """
    results = defaultdict(lambda: defaultdict(list))

    tokenized_docs = tokenized_docs or [doc.split() for doc in docs]
    coherence_dict = corpora.Dictionary(tokenized_docs)
    coherence_corpus = [coherence_dict.doc2bow(tokens) for tokens in tokenized_docs]

    for emb_type, embeddings in embeddings_dict.items():
        if embeddings is None:
            print(f"Skipping {emb_type} due to embedding error.")
            continue
        for min_size in min_size_range:
            try:
                umap_model = umap.UMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                model = BERTopic(
                    min_topic_size=min_size,
                    nr_topics='auto',
                    embedding_model=None,
                    umap_model=umap_model
                )
                topics, probs = model.fit_transform(docs, embeddings=embeddings)
                topic_info = model.get_topic_info()
                num_topics = len(topic_info) - 1

                labels = topics
                silhouette = (
                    silhouette_score(embeddings, labels)
                    if num_topics > 1 and len(set(labels)) > 1
                    else 0
                )

                topics_words = [
                    [word for word, _ in model.get_topic(i)]
                    for i in range(num_topics)
                    if i != -1
                ]
                coherence = 0
                if topics_words:
                    coherence_model = CoherenceModel(
                        topics=topics_words,
                        texts=tokenized_docs,
                        dictionary=coherence_dict,
                        corpus=coherence_corpus,
                        coherence='c_v'
                    )
                    coherence = coherence_model.get_coherence()

                results[emb_type]['Silhouette'].append(silhouette)
                results[emb_type]['Coherence'].append(coherence)
                results[emb_type]['Topic_Count'].append(num_topics)
                results[emb_type]['Min_Size'].append(min_size)

                print(
                    f"Embedding: {emb_type}, Min Topic Size {min_size}: "
                    f"{num_topics} topics, Silhouette: {silhouette:.3f}, Coherence: {coherence:.3f}"
                )

            except Exception as e:
                print(f"Error processing {emb_type} - min_size {min_size}: {e}")
                results[emb_type]['Silhouette'].append(0)
                results[emb_type]['Coherence'].append(0)
                results[emb_type]['Topic_Count'].append(0)
                results[emb_type]['Min_Size'].append(min_size)

    return results


def evaluate_bertopic_single(docs, embeddings, min_topic_size, tokenized_docs, coherence_dict, coherence_corpus):
    """
    One BERTopic fit: returns (num_non_outlier_topics, silhouette, coherence_c_v).
    num_non_outlier_topics matches optimize_bertopic_ablation (len(topic_info) - 1).
    """
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
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topic_info = model.get_topic_info()
    num_topics = len(topic_info) - 1

    labels = topics
    silhouette = (
        silhouette_score(embeddings, labels)
        if num_topics > 1 and len(set(labels)) > 1
        else 0.0
    )

    topics_words = [
        [word for word, _ in model.get_topic(i)]
        for i in range(num_topics)
        if i != -1
    ]
    coherence = 0.0
    if topics_words:
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_docs,
            dictionary=coherence_dict,
            corpus=coherence_corpus,
            coherence="c_v",
        )
        coherence = float(coherence_model.get_coherence())

    return num_topics, silhouette, coherence


def create_ablation_detailed_table(results, embedding_order=None):
    """
    One row per (embedding method, min_topic_size) with coherence (c_v) and silhouette.
    """
    rows = []
    for emb_type, metrics in results.items():
        for i in range(len(metrics["Min_Size"])):
            rows.append(
                {
                    "Embedding": emb_type,
                    "Min_Topic_Size": metrics["Min_Size"][i],
                    "Coherence_Cv": metrics["Coherence"][i],
                    "Silhouette": metrics["Silhouette"][i],
                    "Topic_Count": metrics["Topic_Count"][i],
                }
            )
    df = pd.DataFrame(rows)
    if embedding_order and not df.empty:
        df["Embedding"] = pd.Categorical(
            df["Embedding"], categories=embedding_order, ordered=True
        )
        df = df.sort_values(["Embedding", "Min_Topic_Size"]).reset_index(drop=True)
        df["Embedding"] = df["Embedding"].astype(str)
    return df


def create_ablation_table(results):
    """Create a Pandas DataFrame summarizing the ablation study results."""
    table_data = []
    for emb_type, metrics in results.items():
        best_coherence_idx = np.argmax(metrics['Coherence'])
        best_silhouette_idx = np.argmax(metrics['Silhouette'])

        table_data.append({
            'Embedding': emb_type,
            'Best_Min_Size_Coherence': metrics['Min_Size'][best_coherence_idx],
            'Best_Coherence': metrics['Coherence'][best_coherence_idx],
            'Best_Min_Size_Silhouette': metrics['Min_Size'][best_silhouette_idx],
            'Best_Silhouette': metrics['Silhouette'][best_silhouette_idx],
            'Avg_Topic_Count': np.mean(metrics['Topic_Count'])
        })
    return pd.DataFrame(table_data)


def run_final_bertopic(docs, embeddings, min_topic_size, output_folder):
    """
    Run final BERTopic model with UMAP random_state=42.

    Returns:
        tuple: (model, topics, probs, topic_info)
    """
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    final_model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics='auto',
        embedding_model=None,
        umap_model=umap_model
    )
    final_topics, final_probs = final_model.fit_transform(docs, embeddings=embeddings)
    final_topic_info = final_model.get_topic_info()

    os.makedirs(output_folder, exist_ok=True)
    final_topic_info.to_csv(os.path.join(output_folder, 'final_topic_info.csv'), index=False)

    with open(os.path.join(output_folder, 'topic_details.txt'), 'w') as f:
        f.write("Topic Details:\n\n")
        for topic_id in range(-1, len(final_topic_info) - 1):
            label = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
            if topic_id == -1:
                f.write(f"Topic {topic_id}: {label}\n")
            else:
                topic_words = final_model.get_topic(topic_id)
                if topic_words:
                    words_str = ', '.join([word for word, _ in topic_words])
                    f.write(f"Topic {topic_id}: {label}\n  Top words: {words_str}\n")
            f.write("\n")

    try:
        n_neighbors_umap = min(5, final_model.topic_embeddings_.shape[0] - 1)
        min_dist_umap = 0.0 if final_model.topic_embeddings_.shape[0] < 3 else 0.1
        final_model.topic_embeddings_ = umap.UMAP(
            n_neighbors=n_neighbors_umap,
            n_components=2,
            metric='cosine',
            random_state=42,
            min_dist=min_dist_umap
        ).fit_transform(final_model.topic_embeddings_)
        output_path = os.path.join(output_folder, 'bertopic_final_visualization.html')
        final_model.visualize_topics().write_html(output_path)
        print(f"Final topic visualization saved to {output_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")

    return final_model, final_topics, final_probs, final_topic_info
