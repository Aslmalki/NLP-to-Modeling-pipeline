"""
Thematic comparison module: Model topics vs. human summaries.
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Pre-compiled regex for thematic preprocessing
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'\S+@\S+')
_RE_SPECIAL = re.compile(r'[^a-zA-Z\s]')
_RE_WHITESPACE = re.compile(r'\s+')


def _ensure_nltk_data():
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            if pkg == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif pkg == 'averaged_perceptron_tagger':
                nltk.data.find(f'taggers/{pkg}')
            else:
                nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)


def preprocess_text_for_thematic_analysis(text, custom_stopwords=None):
    """Preprocess text using the same pipeline as the topic modeling."""
    _ensure_nltk_data()
    if custom_stopwords is None:
        custom_stopwords = set(stopwords.words('english'))
        additional_stopwords = {
            'may', 'also', 'one', 'two', 'time', 'result', 'level', 'word',
            'get', 'make', 'use', 'take', 'give', 'find', 'come', 'go',
            'many', 'much', 'several', 'various', 'different',
            'first', 'second', 'third', 'last', 'next', 'previous',
            'year', 'month', 'week', 'day', 'hour', 'minute',
            'etc', 'ie', 'eg', 'et', 'al', 'fig', 'figure', 'table',
            'number', 'example', 'however', 'thus', 'therefore'
        }
        custom_stopwords.update(additional_stopwords)
        domain_terms = {
            'stress', 'adaptation', 'team', 'agent', 'model', 'task', 'group',
            'performance', 'study', 'change', 'desert', 'work', 'social',
            'individual', 'norm', 'research', 'system'
        }
        custom_stopwords = custom_stopwords - domain_terms

    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = _RE_URL.sub('', text)
    text = _RE_EMAIL.sub('', text)
    text = _RE_SPECIAL.sub(' ', text)
    text = _RE_WHITESPACE.sub(' ', text).strip()
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in custom_stopwords and len(token) >= 3
    ]
    return ' '.join(filtered_tokens)


def extract_all_model_topics(model, top_n_words=30):
    """Extract all topics and their top words from the BERTopic model."""
    topic_info = model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1]
    topic_info = topic_info.sort_values('Count', ascending=False)

    topic_words = {}
    for topic_id in topic_info['Topic']:
        topic_data = model.get_topic(topic_id)
        if topic_data:
            words = [word for word, _ in topic_data[:top_n_words]]
            topic_words[topic_id] = words

    return topic_info, topic_words


def extract_human_summary_themes(summary_csv_path, top_n_words=30):
    """Extract key themes from all human summaries combined."""
    try:
        print(f"Loading human summaries from: {summary_csv_path}")
        summaries_df = pd.read_csv(summary_csv_path)
        print(f"Human summaries dataframe shape: {summaries_df.shape}")
        print("Columns in the summaries dataframe:", summaries_df.columns.tolist())

        summary_texts = []
        if 'summary' in summaries_df.columns:
            summary_texts = summaries_df['summary'].fillna('').astype(str).tolist()
        else:
            text_cols = [c for c in summaries_df.columns if 'Discussion' in c or 'Details' in c]
            for _, row in summaries_df.iterrows():
                parts = [
                    str(row[c]).strip()
                    for c in text_cols
                    if c in row.index and pd.notna(row[c]) and str(row[c]).strip()
                ]
                if parts:
                    summary_texts.append(' '.join(parts))

        all_summaries = ' '.join(summary_texts) if summary_texts else ''

        if all_summaries.strip():
            processed_summaries = preprocess_text_for_thematic_analysis(all_summaries)
            try:
                vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
                tfidf_matrix = vectorizer.fit_transform([processed_summaries])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                term_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
                top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
                word_freq = Counter(processed_summaries.split())
                top_freq_words = word_freq.most_common(top_n_words)
                return {
                    'processed_text': processed_summaries,
                    'tfidf_terms': top_terms,
                    'freq_terms': top_freq_words,
                    'all_terms': list(term_scores.keys())
                }
            except Exception as e:
                print(f"Error in TF-IDF extraction: {e}")
                word_freq = Counter(processed_summaries.split())
                top_freq_words = word_freq.most_common(top_n_words)
                return {
                    'processed_text': processed_summaries,
                    'freq_terms': top_freq_words,
                    'all_terms': list(word_freq.keys())
                }
        else:
            print("No summary text found. Ensure CSV has 'summary' or 'Discussion'/'Details' columns.")
            return None
    except Exception as e:
        print(f"Error loading human summaries: {e}")
        return None


def compare_model_and_human_themes(model_topics, human_themes):
    """Compare the themes identified by the model with those from human summaries."""
    all_model_words = set()
    for topic_id, words in model_topics.items():
        all_model_words.update(words)

    all_human_words = set()
    if 'tfidf_terms' in human_themes:
        all_human_words.update([term for term, _ in human_themes['tfidf_terms']])
    if 'freq_terms' in human_themes:
        all_human_words.update([term for term, _ in human_themes['freq_terms']])

    overlapping_terms = all_model_words.intersection(all_human_words)
    unique_to_model = all_model_words - all_human_words
    unique_to_human = all_human_words - all_model_words

    if len(all_model_words) > 0 and len(all_human_words) > 0:
        model_coverage = len(overlapping_terms) / len(all_model_words) * 100
        human_coverage = len(overlapping_terms) / len(all_human_words) * 100
        jaccard_similarity = len(overlapping_terms) / len(all_model_words.union(all_human_words))
    else:
        model_coverage = human_coverage = jaccard_similarity = 0

    topic_similarities = {}
    for topic_id, topic_words_list in model_topics.items():
        topic_word_set = set(topic_words_list)
        overlap = topic_word_set.intersection(all_human_words)
        similarity = len(overlap) / len(topic_word_set) if len(topic_word_set) > 0 else 0
        topic_similarities[topic_id] = {
            'similarity': similarity,
            'overlapping_terms': list(overlap)
        }

    return {
        'overlapping_terms': list(overlapping_terms),
        'unique_to_model': list(unique_to_model),
        'unique_to_human': list(unique_to_human),
        'model_coverage': model_coverage,
        'human_coverage': human_coverage,
        'jaccard_similarity': jaccard_similarity,
        'topic_similarities': topic_similarities,
        'all_model_words': all_model_words,
        'all_human_words': all_human_words
    }


def visualize_thematic_comparison(
    model_info, topic_words, human_themes, comparison_results, output_folder, figures_dir
):
    """Create visualizations under ``figures_dir``; write ``thematic_comparison_report.md`` to ``output_folder``."""
    try:
        from matplotlib_venn import venn2
    except ImportError:
        venn2 = None

    try:
        from wordcloud import WordCloud
    except ImportError:
        WordCloud = None

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    topic_similarities = comparison_results['topic_similarities']
    topic_ids = list(topic_similarities.keys())

    if venn2:
        plt.figure(figsize=(10, 8))
        venn2(
            subsets=(
                len(comparison_results['unique_to_model']),
                len(comparison_results['unique_to_human']),
                len(comparison_results['overlapping_terms'])
            ),
            set_labels=('Model Topics', 'Human Summaries')
        )
        plt.title('Overlap Between Model Topics and Human Summaries', fontsize=16)
        plt.savefig(os.path.join(figures_dir, 'model_human_term_overlap.png'), dpi=300)
        plt.close()

    similarity_values = [topic_similarities[tid]['similarity'] for tid in topic_ids]
    sorted_indices = np.argsort(similarity_values)[::-1]
    sorted_topic_ids = [topic_ids[i] for i in sorted_indices]
    sorted_similarity_values = [similarity_values[i] for i in sorted_indices]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(sorted_topic_ids)), sorted_similarity_values, color='skyblue', alpha=0.7)
    plt.xticks(range(len(sorted_topic_ids)), sorted_topic_ids, rotation=45)
    plt.xlabel('Topic ID', fontsize=14)
    plt.ylabel('Similarity to Human Themes', fontsize=14)
    plt.title('Topic-Level Similarity to Human Themes', fontsize=16)
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{sorted_similarity_values[i]:.2f}',
            ha='center', va='bottom', fontsize=10
        )
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'topic_human_similarity.png'), dpi=300)
    plt.close()

    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    if WordCloud:
        if comparison_results['overlapping_terms']:
            plt.figure(figsize=(12, 8))
            wc = WordCloud(
                width=800, height=500, background_color='white',
                colormap=mcolors.ListedColormap(colors), max_words=100,
                contour_width=1, contour_color='steelblue'
            ).generate(' '.join(comparison_results['overlapping_terms']))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Terms Found in Both Model Topics and Human Summaries', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'overlapping_terms_wordcloud.png'), dpi=300)
            plt.close()

        if comparison_results['unique_to_model']:
            plt.figure(figsize=(12, 8))
            wc = WordCloud(
                width=800, height=500, background_color='white',
                colormap=mcolors.ListedColormap([colors[0]]), max_words=100,
                contour_width=1, contour_color='steelblue'
            ).generate(' '.join(comparison_results['unique_to_model']))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Terms Unique to Model Topics', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'model_unique_terms_wordcloud.png'), dpi=300)
            plt.close()

        if comparison_results['unique_to_human']:
            plt.figure(figsize=(12, 8))
            wc = WordCloud(
                width=800, height=500, background_color='white',
                colormap=mcolors.ListedColormap([colors[1]]), max_words=100,
                contour_width=1, contour_color='steelblue'
            ).generate(' '.join(comparison_results['unique_to_human']))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Terms Unique to Human Summaries', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'human_unique_terms_wordcloud.png'), dpi=300)
            plt.close()

    top_overlapping = (
        comparison_results['overlapping_terms'][:30]
        if len(comparison_results['overlapping_terms']) > 30
        else comparison_results['overlapping_terms']
    )
    if top_overlapping and topic_ids:
        term_topic_matrix = np.zeros((len(top_overlapping), len(topic_ids)))
        for i, term in enumerate(top_overlapping):
            for j, tid in enumerate(topic_ids):
                tw_set = set(topic_words.get(tid, []))
                term_topic_matrix[i, j] = 1 if term in tw_set else 0
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            term_topic_matrix, cmap='YlGnBu', cbar=False,
            linewidths=0.5, linecolor='gray',
            xticklabels=topic_ids, yticklabels=top_overlapping
        )
        plt.title('Presence of Key Shared Terms Across Topics', fontsize=16)
        plt.xlabel('Topic ID', fontsize=14)
        plt.ylabel('Term', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'term_topic_heatmap.png'), dpi=300)
        plt.close()

    all_model_words = comparison_results['all_model_words']
    all_human_words = comparison_results['all_human_words']
    report = f"""
# Thematic Comparison Report: Model Topics vs. Human Summaries

## Overall Similarity Metrics

- **Jaccard Similarity**: {comparison_results['jaccard_similarity']:.4f}
- **Model Coverage**: {comparison_results['model_coverage']:.2f}% of model terms appear in human summaries
- **Human Coverage**: {comparison_results['human_coverage']:.2f}% of human terms appear in model topics

## Term Analysis

- **Total terms in model topics**: {len(all_model_words)}
- **Total terms in human summaries**: {len(all_human_words)}
- **Overlapping terms**: {len(comparison_results['overlapping_terms'])}
- **Terms unique to model**: {len(comparison_results['unique_to_model'])}
- **Terms unique to human summaries**: {len(comparison_results['unique_to_human'])}

### Top 20 Overlapping Terms

{', '.join(comparison_results['overlapping_terms'][:20])}

### Top 20 Terms Unique to Model

{', '.join(comparison_results['unique_to_model'][:20])}

### Top 20 Terms Unique to Human Summaries

{', '.join(comparison_results['unique_to_human'][:20])}

## Topic-by-Topic Analysis

"""
    for tid in sorted_topic_ids:
        ts = topic_similarities[tid]
        report += f"""
### Topic {tid}

- **Similarity to human themes**: {ts['similarity']:.4f}
- **Overlapping terms**: {', '.join(ts['overlapping_terms'][:10])}

"""
    report += """
## Research Implications

The comparison between automated topic modeling and human thematic analysis reveals several important insights:

1. **Conceptual Alignment**: The degree of overlap indicates how well the automated approach captures human-recognizable patterns.
2. **Blind Spots**: Terms unique to human summaries may represent conceptual understanding that statistical methods don't capture.
3. **Topic Relevance**: Topics with higher similarity to human themes may be more interpretable.
4. **Validation Strategy**: This comparison provides a robust validation approach for exploratory research.

## Recommendations

1. Focusing on topics with higher human theme similarity for primary research insights
2. Exploring unique model terms as potential novel patterns
3. Incorporating human-unique terms into future model iterations
4. Using this thematic comparison as a standard validation step in mixed-methods research
"""
    with open(os.path.join(output_folder, 'thematic_comparison_report.md'), 'w') as f:
        f.write(report)
    print(f"Thematic comparison report saved to {os.path.join(output_folder, 'thematic_comparison_report.md')}")

    return {
        'jaccard_similarity': comparison_results['jaccard_similarity'],
        'model_coverage': comparison_results['model_coverage'],
        'human_coverage': comparison_results['human_coverage'],
        'top_topics_by_similarity': sorted_topic_ids[:5]
    }


def run_thematic_comparison(model, summary_csv_path, output_folder, figures_dir=None):
    """Run the complete thematic comparison pipeline. Figures go to ``figures_dir`` (default: config FIGURES_DIR)."""
    from config import FIGURES_DIR as _FIGURES_DIR

    if figures_dir is None:
        figures_dir = _FIGURES_DIR
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Extracting model topics...")
    topic_info, topic_words = extract_all_model_topics(model)

    print("Extracting human summary themes...")
    human_themes = extract_human_summary_themes(summary_csv_path)

    if human_themes:
        print("Comparing model topics with human themes...")
        comparison_results = compare_model_and_human_themes(topic_words, human_themes)

        print("Creating visualizations...")
        summary = visualize_thematic_comparison(
            topic_info, topic_words, human_themes, comparison_results, output_folder, figures_dir
        )

        print("\nThematic comparison complete!")
        print(f"Jaccard similarity: {summary['jaccard_similarity']:.4f}")
        print(f"Model coverage: {summary['model_coverage']:.2f}%")
        print(f"Human coverage: {summary['human_coverage']:.2f}%")
        print(f"Top topics by similarity: {summary['top_topics_by_similarity']}")
        return summary
    else:
        print("Could not extract human summary themes. Please check the CSV file.")
        return None
