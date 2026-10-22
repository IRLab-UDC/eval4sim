# Distance-based authorship verification baseline and evaluation metrics of the
# PAN 2023 authorship verification task (Bevendorff et al., 2023).
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return 0.0


def c_at_1(y, p):
    answered = p != 0.5
    correct = np.sum(answered & ((p > 0.5) == (y > 0.5)))
    unanswered = np.sum(~answered)
    return (correct + unanswered * correct / len(p)) / len(p)


def f1(y, p):
    answered = p != 0.5
    return f1_score(y[answered], (p[answered] >= 0.5).astype(float), zero_division=0.0)


def f_05_u(y, p):
    tp = np.sum((p > 0.5) & (y == 1))
    fp = np.sum((p > 0.5) & (y == 0))
    fn = np.sum((p < 0.5) & (y == 1))
    unanswered = np.sum(p == 0.5)
    return 1.25 * tp / (1.25 * tp + 0.25 * (fn + unanswered) + fp)


def brier(y, p):
    try:
        return 1 - brier_score_loss(y, p)
    except ValueError:
        return 0.0


def evaluate_all(y, p):
    results = {'auc': auc(y, p), 'c@1': c_at_1(y, p), 'f_05_u': f_05_u(y, p), 'F1': f1(y, p), 'brier': brier(y, p)}
    results['overall'] = np.mean(list(results.values()))
    return {metric: round(float(value), 3) for metric, value in results.items()}


def calibrate(similarities, p1, p2):
    low = similarities / p1 * 0.49
    high = 0.51 + (similarities - p2) / (1 - p2) * 0.49
    return np.where(similarities <= p1, low, np.where(similarities < p2, 0.5, high))


def cosine(a, b):
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), norms, out=np.zeros_like(norms), where=norms > 0)


def similarities(model, pairs):
    x1 = model['vectorizer'].transform([a for a, _ in pairs]).toarray()
    x2 = model['vectorizer'].transform([b for _, b in pairs]).toarray()
    return np.mean([cosine(x1[:, features], x2[:, features]) for features in model['features']], axis=0)


def fit(pairs, labels, vocab_size=3000, ngram_size=4, num_iterations=100, dropout=0.5, seed=2020):
    rng = np.random.RandomState(seed)
    vectorizer = TfidfVectorizer(max_features=vocab_size, analyzer='char', ngram_range=(ngram_size, ngram_size))
    vectorizer.fit([text for pair in pairs for text in pair])
    total = len(vectorizer.get_feature_names_out())
    model = {
        'vectorizer': vectorizer,
        'features': [rng.choice(total, int(total * dropout), replace=False) for _ in range(num_iterations)],
    }

    sims = similarities(model, pairs)
    thresholds = np.arange(0.01, 0.99, 0.01)
    model['p1'], model['p2'] = max(
        combinations(thresholds, 2),
        key=lambda p: evaluate_all(labels, calibrate(sims, *p))['overall'],
    )
    return model


def predict(model, pairs):
    return calibrate(similarities(model, pairs), model['p1'], model['p2'])
