<div align="center">

# 🧠 Word2Vec From Scratch

**A minimal, dependency-light implementation of Skip-gram with Negative Sampling — built with just NumPy.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat&logo=numpy&logoColor=white)
![No Frameworks](https://img.shields.io/badge/No%20PyTorch%2FTensorFlow-required-red?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-educational-yellow?style=flat)

</div>

No PyTorch, no TensorFlow, no `gensim`. Every gradient is derived and applied by hand.

This project exists to answer one question: *what actually happens inside a word embedding model?* Rather than importing a pretrained embedding or calling a library's `Word2Vec()` constructor, this repo builds the whole pipeline — tokenization, training pair generation, negative sampling, gradient updates, and evaluation — from first principles.

## Demo Results

Trained on a small hand-designed corpus (~700 words), the model correctly separates semantically related word pairs from unrelated ones using nothing but cosine similarity on learned vectors:

| Word Pair | Relationship | Cosine Similarity |
|---|---|---|
| cat + dog | 🟢 related (animals) | +0.53 |
| doctor + nurse | 🟢 related (profession) | +0.68 |
| apple + banana | 🟢 related (fruit) | +0.68 |
| car + bike | 🟢 related (vehicle) | +0.57 |
| king + queen | 🟢 related (royalty) | +0.18 |
| cat + king | 🔴 unrelated | -0.12 |
| king + bike | 🔴 unrelated | +0.00 |

Related pairs cluster meaningfully above zero; unrelated pairs sit near zero or negative — confirming the model is learning real distributional structure, not noise.

## How It Works

### 1. Corpus & Preprocessing
Text is lowercased and stripped of punctuation, then tokenized on whitespace. A vocabulary is built from the unique token set, with `word_to_idx` / `idx_to_word` lookup tables.

Corpus design matters a lot at this scale — see [Design Notes](#design-notes) below.

### 2. Training Pair Generation
For every word in the corpus (the "center" word), all words within a fixed context window (`WINDOW_SIZE`) are extracted as **positive** `(center, context)` pairs — the classic skip-gram formulation.

### 3. Embeddings
Each vocabulary word is assigned a randomly initialized vector (`EMBEDDING_DIM`-dimensional, small-scale Gaussian init). These vectors *are* the model — there's no separate weight matrix beyond them in this simplified single-vector-per-word setup.

### 4. Negative Sampling
For every positive pair, a small number of "negative" (fake) pairs are sampled — random words that did *not* appear in that context — and the model is trained to push their similarity **down**, while pushing the real pair's similarity **up**.

Negative words are sampled from a **frequency-weighted distribution** (unigram frequency raised to the 3/4 power), not uniformly. Uniform sampling over-selects extremely common words and drowns out the meaningful negative signal — this was one of the more important implementation details to get right.

### 5. Training
- Loss: binary logistic loss (sigmoid + negative log-likelihood), applied separately to positive and negative samples
- Optimizer: plain SGD with manually derived gradients (no autograd)
- Learning rate decay applied linearly across epochs to prevent late-stage oscillation on a small, noisy dataset

### 6. Evaluation
Cosine similarity is computed between trained vectors for hand-picked related and unrelated word pairs, used as a sanity check that the model learned genuine semantic structure rather than memorizing noise.

### 7. Visualization
Vectors are L2-normalized (to remove magnitude effects unrelated to cosine similarity) and projected to 2D using **t-SNE** rather than PCA. On a small vocabulary, t-SNE's local-neighborhood preservation shows semantic clusters far more clearly than PCA's linear, global-variance approach.

## Design Notes

A few non-obvious lessons from building this:

- **Corpus repetition matters more than corpus size.** A small, naturally-varied corpus barely trains at all — each word only co-occurs with its neighbors once or twice, giving the model almost no signal. Rebuilding the corpus around **repeated sentence templates across related words** (e.g. "the king wears a crown" / "the queen wears a crown") produced dramatically better clustering with the same total word count.
- **Negative sampling distribution is not a minor detail.** Switching from uniform to frequency-weighted (`freq^0.75`) sampling was one of the single biggest quality improvements in this project.
- **Embedding dimensionality should scale with vocabulary size.** For a ~150–200 word vocabulary, a large embedding dimension (e.g. 42+) is under-constrained and produces noisier low-dimensional projections. A smaller dimension (~10) forces more meaningful compression.
- **PCA vs. t-SNE is not just a style choice.** The same trained vectors can look diffuse under PCA and clearly clustered under t-SNE — projection choice changes what story the visualization tells.

## Usage

```bash
pip install numpy matplotlib scikit-learn seaborn

python word2vec.py
```

This will:
1. Tokenize the corpus and build the vocabulary
2. Generate skip-gram training pairs
3. Train embeddings via SGNS for `NUM_EPOCHS` epochs
4. Print cosine similarity for a set of related/unrelated word pairs
5. Save a t-SNE cluster plot and similarity heatmap to disk

## What This Is Not

This is an educational implementation, not a production-grade or performance-optimized one. It intentionally avoids:
- Hierarchical softmax or other large-vocabulary optimizations
- Subword/character-level tokenization
- Batched matrix operations across all pairs simultaneously
- Pretrained embeddings or transfer learning

For production use cases, `gensim`, `spaCy`, or transformer-based embeddings (e.g. `sentence-transformers`) are the right tool.

