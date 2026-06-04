"""
Visualization and table generation for NLP Topic Modeling Pipeline.
Addresses PLOS One reviewer expectations: ablation viz, document distribution,
improved co-occurrence networks, corpus/topic/thematic tables.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from config import TOPIC_LABELS, FIGURES_DIR


def _ensure_colorblind_friendly():
    """Use colorblind-friendly palette if available."""
    try:
        sns.set_palette("colorblind")
    except Exception:
        pass


def create_ablation_visualization(ablation_results, figures_dir=None):
    """
    Create grouped bar chart: Coherence and Silhouette by embedding type and min_topic_size.
    Addresses reviewer expectation: visual justification for model choice.
    """
    _ensure_colorblind_friendly()
    rows = []
    for emb_type, metrics in ablation_results.items():
        for i in range(len(metrics['Min_Size'])):
            rows.append({
                'Embedding': emb_type,
                'Min_Topic_Size': metrics['Min_Size'][i],
                'Coherence': metrics['Coherence'][i],
                'Silhouette': metrics['Silhouette'][i],
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coherence by embedding and min_size
    pivot_coherence = df.pivot_table(
        index='Min_Topic_Size', columns='Embedding', values='Coherence'
    )
    pivot_coherence.plot(kind='bar', ax=axes[0], width=0.8, rot=0)
    axes[0].set_xlabel('Minimum Topic Size', fontsize=12)
    axes[0].set_ylabel('Coherence (c_v)', fontsize=12)
    axes[0].set_title('Topic Coherence by Embedding and Min Topic Size', fontsize=14)
    axes[0].legend(title='Embedding', bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Silhouette by embedding and min_size
    pivot_silhouette = df.pivot_table(
        index='Min_Topic_Size', columns='Embedding', values='Silhouette'
    )
    pivot_silhouette.plot(kind='bar', ax=axes[1], width=0.8, rot=0)
    axes[1].set_xlabel('Minimum Topic Size', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score by Embedding and Min Topic Size', fontsize=14)
    axes[1].legend(title='Embedding', bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    figures_dir = figures_dir or FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, 'ablation_study_results.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation visualization saved to {path}")


def create_document_distribution_chart(final_topics, topic_info, figures_dir=None):
    """
    Bar chart: document count per topic. Shows balance vs. outlier dominance.
    """
    _ensure_colorblind_friendly()
    topic_counts = topic_info.set_index('Topic')['Count'].to_dict()
    topic_ids = [t for t in sorted(topic_counts.keys()) if t != -1]
    topic_ids = [-1] + topic_ids  # Outliers first
    labels = [f"Topic {t}" if t != -1 else "Outliers" for t in topic_ids]
    counts = [topic_counts.get(t, 0) for t in topic_ids]
    colors = ['#999999' if t == -1 else None for t in topic_ids]
    if not any(colors):
        colors = None
    else:
        colors = [plt.cm.Set3(i % 12) if t != -1 else '#999999'
                  for i, t in enumerate(topic_ids)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(topic_ids)), counts, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(range(len(topic_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title('Document Distribution Across Topics', fontsize=14)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(c), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    figures_dir = figures_dir or FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, 'document_distribution_by_topic.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Document distribution chart saved to {path}")


def create_corpus_statistics_table(preprocessed_docs, word_counts, output_folder):
    """Create Table 1: Corpus statistics for Methods section."""
    n_docs = len(preprocessed_docs)
    vocab_size = len(word_counts)
    total_tokens = sum(word_counts.values())
    doc_lengths = [len(doc.split()) for doc in preprocessed_docs]
    mean_len = np.mean(doc_lengths)
    max_len = np.max(doc_lengths)
    min_len = np.min(doc_lengths)
    std_len = np.std(doc_lengths)

    table = pd.DataFrame([
        {'Metric': 'Number of documents', 'Value': n_docs},
        {'Metric': 'Vocabulary size (unique tokens)', 'Value': vocab_size},
        {'Metric': 'Total tokens', 'Value': total_tokens},
        {'Metric': 'Mean document length (tokens)', 'Value': f'{mean_len:.1f}'},
        {'Metric': 'Max document length (tokens)', 'Value': max_len},
        {'Metric': 'Min document length (tokens)', 'Value': min_len},
        {'Metric': 'Std document length', 'Value': f'{std_len:.1f}'},
    ])
    path = os.path.join(output_folder, 'table1_corpus_statistics.csv')
    table.to_csv(path, index=False)
    print(f"Corpus statistics table saved to {path}")
    return table


def create_topic_interpretation_table(final_model, topic_info, output_folder):
    """
    Create Table 3: Topic interpretation with labels and Top 2 context.
    """
    rows = []
    total_docs = topic_info['Count'].sum()
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        count = row['Count']
        pct = 100 * count / total_docs if total_docs > 0 else 0
        label = TOPIC_LABELS.get(tid, f"Topic {tid}")
        topic_data = final_model.get_topic(tid)
        top_words = ', '.join([w for w, _ in topic_data[:10]]) if topic_data else ''
        rows.append({
            'Topic': tid,
            'Label': label,
            'Top Words': top_words,
            'Document Count': count,
            'Percent (%)': f'{pct:.1f}',
        })
    table = pd.DataFrame(rows)
    path = os.path.join(output_folder, 'table3_topic_interpretation.csv')
    table.to_csv(path, index=False)
    print(f"Topic interpretation table saved to {path}")

    # Also save Topic 2 context as supplementary note
    topic2_note = """Topic 2: Specialized Remote Sensing and Imaging Hardware — Interpretation Note

This topic clusters terms related to instrumentation used in extreme environments:
- GVD: Baikal-GVD (Gigaton Volume Detector), neutrino telescope in Lake Baikal
- MSFC: Marshall Space Flight Center (NASA)
- Watec/Sony/Stellacam: Low-light CCD/video cameras for meteor/astronomy observation
- CASS: Center for Astrophysics & Space Sciences or Cassini-related
- Alamo: Air-Launched Autonomous Micro-Observer floats (oceanography)
- Ult: Ultra-Low Temperature or Ultraviolet imaging
- Wax/Schm: Optical designs (e.g., Schmidt camera) or physical phenomena (Schumann resonances)

The topic bridges the "where" (extreme environments) and the "how" (tools for observation).
"""
    with open(os.path.join(output_folder, 'topic2_interpretation_note.txt'), 'w') as f:
        f.write(topic2_note)
    return table


def create_thematic_comparison_table(comparison_summary, output_folder):
    """Create Table 4: Thematic comparison metrics."""
    if comparison_summary is None:
        return
    table = pd.DataFrame([
        {'Metric': 'Jaccard Similarity', 'Value': f"{comparison_summary['jaccard_similarity']:.4f}"},
        {'Metric': 'Model Coverage (%)', 'Value': f"{comparison_summary['model_coverage']:.2f}"},
        {'Metric': 'Human Coverage (%)', 'Value': f"{comparison_summary['human_coverage']:.2f}"},
    ])
    path = os.path.join(output_folder, 'table4_thematic_comparison.csv')
    table.to_csv(path, index=False)
    print(f"Thematic comparison table saved to {path}")


def draw_improved_cooccurrence_network(G, title, output_path, top_n_edges=75):
    """
    Draw co-occurrence network with improved layout and readability.
    - Subgraph of top N edges by weight
    - Node size by degree
    - Edge width by weight
    - Force-directed layout
    """
    if G.number_of_edges() == 0:
        print(f"Skipping {title}: no edges")
        return

    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
    top_edges = edges_sorted[:top_n_edges]
    H = nx.Graph()
    for u, v, d in top_edges:
        H.add_edge(u, v, **d)

    pos = nx.spring_layout(H, k=1.5, iterations=50, seed=42)
    degrees = dict(H.degree())
    node_sizes = [300 + 100 * degrees.get(n, 0) for n in H.nodes()]
    edge_weights = [H[u][v].get('weight', 1) for u, v in H.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 3 * (w / max_w) for w in edge_weights]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color='lightsteelblue',
                           edgecolors='gray', linewidths=0.5)
    nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(H, pos, font_size=8, font_weight='bold')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
