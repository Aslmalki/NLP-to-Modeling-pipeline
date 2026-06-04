"""
Embedding generation for NLP Topic Modeling Pipeline.
ODE-compliant: Doc2Vec seed parameter for reproducibility.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def get_tfidf_embedding(docs):
    """Generate TF-IDF embeddings."""
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    embeddings = vectorizer.fit_transform(docs).toarray()
    return embeddings, vectorizer.get_feature_names_out()


def get_doc2vec_embedding(
    docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    epochs=20,
    seed=42,
    doc_keys=None,
):
    """Generate Doc2Vec embeddings with seed parameter for reproducibility.

    workers=1 ensures deterministic training with a fixed seed (multi-worker training is nondeterministic).
    If doc_keys is provided (e.g. Paper ID strings), tags use those so vectors are keyed by stable ids
    for validation-only runs and infer_vector on new documents.
    """
    tokenized_docs = [doc.split() for doc in docs]
    if doc_keys is not None:
        if len(doc_keys) != len(docs):
            raise ValueError("doc_keys must have the same length as docs")
        tags = [str(k) for k in doc_keys]
    else:
        tags = [str(i) for i in range(len(docs))]
    tagged_data = [
        TaggedDocument(words=tokens, tags=[tags[i]])
        for i, tokens in enumerate(tokenized_docs)
    ]
    model = Doc2Vec(
        tagged_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed
    )
    embeddings = np.array([model.dv[tags[i]] for i in range(len(docs))])
    return embeddings, model


def get_bert_embedding(docs, model_name='allenai/scibert_scivocab_uncased'):
    """Generate BERT embeddings with batch processing. Uses GPU if available, else CPU."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # use_safetensors=True avoids torch.load vulnerability (CVE-2025-32434) when torch < 2.6
        model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        batch_size = 16
        embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            doc_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(doc_embeddings)

        return np.vstack(embeddings), list(tokenizer.get_vocab().keys())
    except (OSError, RuntimeError) as e:
        err_msg = str(e).lower()
        if "is not a local folder and is not a valid model identifier" in str(e):
            print(f"Warning: Model '{model_name}' not found on Hugging Face Model Hub. Skipping.")
        elif "no space left" in err_msg or "disk" in err_msg or "os error 28" in err_msg:
            print(f"Warning: Insufficient disk space to download '{model_name}'. Skipping BERT.")
        else:
            print(f"Warning: Could not load '{model_name}': {e}. Skipping.")
        return None, None
