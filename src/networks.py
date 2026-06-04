"""
Network analysis module for NLP Topic Modeling Pipeline.
Co-occurrence networks at sentence and document level.
"""
from collections import defaultdict, Counter
from itertools import combinations, chain

import networkx as nx


def build_sentence_cooccurrence(docs, top_n=500, max_vocab_words=1000):
    """Build a co-occurrence network at the sentence level. Limits vocabulary to top frequent words."""
    all_words = list(chain.from_iterable(doc.split() for doc in docs))
    word_freq = Counter(all_words)
    vocab = set(w for w, _ in word_freq.most_common(max_vocab_words))

    cooccur_dict = defaultdict(int)
    for doc in docs:
        sentences = doc.split('. ')
        for sentence in sentences:
            words = [w for w in sentence.split() if w in vocab]
            for w1, w2 in combinations(words, 2):
                if w1 != w2:
                    cooccur_dict[(w1, w2)] += 1

    G = nx.Graph()
    top_pairs = sorted(cooccur_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for (w1, w2), count in top_pairs:
        G.add_edge(w1, w2, weight=count)
    return G


def build_doc_cooccurrence(docs, top_n=500, max_vocab_words=1000):
    """Build a co-occurrence network at the document level. Limits vocabulary to top frequent words."""
    all_words = list(chain.from_iterable(doc.split() for doc in docs))
    word_freq = Counter(all_words)
    vocab = set(w for w, _ in word_freq.most_common(max_vocab_words))

    cooccur_dict = defaultdict(int)
    for doc in docs:
        words = [w for w in doc.split() if w in vocab]
        words = set(words)
        for w1, w2 in combinations(words, 2):
            if w1 != w2:
                cooccur_dict[(w1, w2)] += 1

    G = nx.Graph()
    top_pairs = sorted(cooccur_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for (w1, w2), count in top_pairs:
        G.add_edge(w1, w2, weight=count)
    return G
