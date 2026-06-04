"""
NLP Topic Modeling Pipeline - Human Behavior in Extreme Environments.
ODE-compliant: set_all_seeds at start, config paths throughout.
"""
import os
import sys

# Repository root: ``import config`` and ``src`` resolve from here
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from config import (
    check_torch_version,
    set_all_seeds,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    FIGURES_DIR,
    HUMAN_ANNOTATION_CSV,
    MIN_SIZE_RANGE,
    MAX_VOCAB_WORDS,
    DOC2VEC_SEED,
    TOP_EDGES_FOR_VIZ,
)

from src.preprocessing import (
    get_custom_stopwords,
    load_data_from_drive,
    preprocess_text,
    calculate_dataset_statistics,
)
from src.embeddings import (
    get_tfidf_embedding,
    get_doc2vec_embedding,
    get_bert_embedding,
)
import numpy as np

from src.topic_modeling import (
    optimize_bertopic_ablation,
    create_ablation_table,
    create_ablation_detailed_table,
    run_final_bertopic,
)
from src.validation import run_thematic_comparison
from src.networks import build_sentence_cooccurrence, build_doc_cooccurrence
from src.visualizations import (
    create_ablation_visualization,
    create_document_distribution_chart,
    create_corpus_statistics_table,
    create_topic_interpretation_table,
    create_thematic_comparison_table,
    draw_improved_cooccurrence_network,
)

# Set False to skip Steps 6–7 (kappa path only; faster, no extra paper figures).
ENABLE_OPTIONAL_STEPS = True


def main():
    # Check PyTorch version before loading BERT models
    check_torch_version()
    # ODE fix: Set all seeds at pipeline start
    set_all_seeds()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load and preprocess
    print("Loading data...")
    docs, filenames = load_data_from_drive(INPUT_FOLDER)
    custom_stopwords = get_custom_stopwords()
    preprocessed_docs = preprocess_text(docs, min_word_length=3, custom_stopwords=custom_stopwords)
    print(f"Loaded and preprocessed {len(preprocessed_docs)} documents from {INPUT_FOLDER}")

    word_counts, top_words = calculate_dataset_statistics(preprocessed_docs)
    with open(os.path.join(OUTPUT_FOLDER, 'top_frequent_words.txt'), 'w') as f:
        f.write("Top 20 most frequent words:\n")
        for word, count in top_words:
            f.write(f"{word}: {count}\n")
    print(f"\nTop frequent words saved to {os.path.join(OUTPUT_FOLDER, 'top_frequent_words.txt')}")

    # 2. Generate embeddings (Doc2Vec with seed=42)
    print("\nGenerating embeddings...")
    embeddings_tfidf, _ = get_tfidf_embedding(preprocessed_docs)
    embeddings_doc2vec, _ = get_doc2vec_embedding(preprocessed_docs, seed=DOC2VEC_SEED)
    embeddings_scibert, _ = get_bert_embedding(preprocessed_docs, model_name='allenai/scibert_scivocab_uncased')
    embeddings_socialbert, _ = get_bert_embedding(preprocessed_docs, model_name='ESGBERT/SocialBERT-social')

    embeddings_dict = {
        'TFIDF': embeddings_tfidf,
        'Doc2Vec': embeddings_doc2vec,
        'SciBERT': embeddings_scibert,
        'SocialBERT': embeddings_socialbert,
    }
    # Top2Vec document_vectors are used as embeddings only. All configurations use
    # BERTopic UMAP and HDBSCAN for clustering so results are directly comparable
    # across all five embedding methods. (PyPI Top2Vec has no seed= kwarg; use set_all_seeds above.)
    try:
        from top2vec import Top2Vec

        # Top2Vec PyPI build has no seed= kwarg; global seeds set via set_all_seeds(42) above.
        # embedding_model='doc2vec' required for speed='learn' (see Top2Vec speed parameter).
        top2vec_model = Top2Vec(
            documents=preprocessed_docs,
            embedding_model="doc2vec",
            min_count=2,
            speed="learn",
            workers=1,
        )
        doc_vectors = top2vec_model.document_vectors
        embeddings_dict['Top2Vec'] = np.asarray(doc_vectors)
        print(
            f"Top2Vec: document_vectors shape {embeddings_dict['Top2Vec'].shape}"
        )
    except Exception as e:
        print(f"Top2Vec skipped: {e}")

    embeddings_dict = {k: v for k, v in embeddings_dict.items() if v is not None}
    print(f"Generated embeddings for {len(embeddings_dict)} models: {', '.join(embeddings_dict.keys())}")

    # 3. Topic modeling ablation and final model
    print("\nRunning topic modeling optimization...")
    ablation_results = optimize_bertopic_ablation(
        preprocessed_docs, embeddings_dict, MIN_SIZE_RANGE
    )
    ablation_table = create_ablation_table(ablation_results)
    print("\nAblation Study Results (summary per embedding):")
    print(ablation_table)
    ablation_table.to_csv(os.path.join(OUTPUT_FOLDER, 'ablation_study_results.csv'), index=False)
    ablation_table.to_csv(os.path.join(OUTPUT_FOLDER, 'table2_ablation_results.csv'), index=False)

    ablation_order = ['TFIDF', 'Doc2Vec', 'SciBERT', 'SocialBERT', 'Top2Vec']
    ablation_detailed = create_ablation_detailed_table(
        ablation_results, embedding_order=[e for e in ablation_order if e in ablation_results]
    )
    ablation_csv = os.path.join(OUTPUT_FOLDER, 'ablation_study_results_with_top2vec.csv')
    ablation_detailed.to_csv(ablation_csv, index=False)
    print(f"\nFull ablation grid saved to {ablation_csv}")
    print("\nFull ablation grid (all combinations: coherence c_v & silhouette):")
    print(ablation_detailed.to_string(index=False))

    best_embedding_type = ablation_table.loc[ablation_table['Best_Coherence'].idxmax(), 'Embedding']
    optimal_size = int(ablation_table.loc[ablation_table['Best_Coherence'].idxmax(), 'Best_Min_Size_Coherence'])
    final_embeddings = embeddings_dict[best_embedding_type]

    print(f"\nRunning final BERTopic with {best_embedding_type} and min_topic_size={optimal_size}...")
    final_model, final_topics, final_probs, final_topic_info = run_final_bertopic(
        preprocessed_docs, final_embeddings, optimal_size, OUTPUT_FOLDER
    )
    print("\nFinal BERTopic Results:")
    print(final_topic_info)

    # 4. Thematic comparison / validation (kappa path; when human annotation CSV exists)
    comparison_summary = None
    if os.path.exists(HUMAN_ANNOTATION_CSV):
        print("\nRunning thematic comparison...")
        comparison_summary = run_thematic_comparison(
            final_model, HUMAN_ANNOTATION_CSV, OUTPUT_FOLDER, FIGURES_DIR
        )
    else:
        print(f"\nHuman annotation CSV not found at {HUMAN_ANNOTATION_CSV}, skipping thematic comparison.")

    if ENABLE_OPTIONAL_STEPS:
        # Step 6 (optional): Generate network graph
        # Not required to reproduce the kappa result but needed for full paper figures.
        print("\n[Step 6 optional] Building co-occurrence networks...")
        sentence_network = build_sentence_cooccurrence(
            preprocessed_docs, max_vocab_words=MAX_VOCAB_WORDS
        )
        doc_network = build_doc_cooccurrence(
            preprocessed_docs, max_vocab_words=MAX_VOCAB_WORDS
        )

        draw_improved_cooccurrence_network(
            sentence_network,
            "Sentence-Level Co-occurrence Network (Top 75 edges)",
            os.path.join(FIGURES_DIR, 'sentence_cooccurrence_network.png'),
            top_n_edges=TOP_EDGES_FOR_VIZ,
        )
        draw_improved_cooccurrence_network(
            doc_network,
            "Document-Level Co-occurrence Network (Top 75 edges)",
            os.path.join(FIGURES_DIR, 'doc_cooccurrence_network.png'),
            top_n_edges=TOP_EDGES_FOR_VIZ,
        )

        with open(os.path.join(OUTPUT_FOLDER, 'cooccurrence_networks.txt'), 'w') as f:
            f.write("Sentence-Level:\n")
            for u, v, d in sentence_network.edges(data=True):
                f.write(f"{u} - {v}: {d['weight']}\n")
            f.write("\nDocument-Level:\n")
            for u, v, d in doc_network.edges(data=True):
                f.write(f"{u} - {v}: {d['weight']}\n")
        print(f"Networks saved to {os.path.join(OUTPUT_FOLDER, 'cooccurrence_networks.txt')}")

        # Step 7 (optional): Generate all figures
        # Not required to reproduce the kappa result but needed to fully reproduce all paper figures.
        print("\n[Step 7 optional] Generating all figures...")
        create_ablation_visualization(ablation_results)
        create_corpus_statistics_table(preprocessed_docs, word_counts, OUTPUT_FOLDER)
        create_document_distribution_chart(final_topics, final_topic_info)
        create_topic_interpretation_table(final_model, final_topic_info, OUTPUT_FOLDER)
        if comparison_summary is not None:
            create_thematic_comparison_table(comparison_summary, OUTPUT_FOLDER)
    else:
        print("\nSkipping optional Steps 6–7 (set ENABLE_OPTIONAL_STEPS=True for full paper figures).")

    print("\nPipeline complete. Outputs saved to", OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
