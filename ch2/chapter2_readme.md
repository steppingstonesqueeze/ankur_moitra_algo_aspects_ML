# Chapter 2: Nonnegative Matrix Factorization

## 📚 Overview

This chapter explores **Nonnegative Matrix Factorization (NMF)** - a powerful technique for finding interpretable, non-negative factors in data matrices. We journey from the limitations of SVD to polynomial-time algorithms for the separable case, culminating in practical topic modeling applications.

### 🎯 Key Learning Objectives
- Understand why **SVD fails** for interpretable topic modeling
- Grasp the **geometric intuition** behind NMF (cones, separability)
- Master the **Anchor Words Algorithm** for polynomial-time separable NMF
- Apply these concepts to **real topic modeling** problems

---

## 🗂️ Chapter Structure

### Section 2.1: Introduction & Motivation
**Why do we need NMF when SVD exists?**

**Key Concepts:**
- SVD review: M = UΣV^T
- Eckart-Young theorem (optimal rank-k approximation)  
- Frobenius norm and rotational invariance
- **Problem 1**: Orthogonal topics (real topics overlap!)
- **Problem 2**: Negative values (no interpretation)

**Interactive Demo:** [SVD Fundamentals Demo](./svd_fundamentals_demo.html)
- Compare SVD vs NMF on document similarity
- Visualize the "negative contribution" problem
- Understand why we need non-negative constraints

**Code Implementation:**
```python
# SVD for LSI (Latent Semantic Indexing)
def latent_semantic_indexing(term_doc_matrix, k=10):
    U, sigma, Vt = svd(term_doc_matrix)
    # Project documents into k-dimensional topic space
    doc_embeddings = U[:, :k].T @ term_doc_matrix
    return doc_embeddings, U[:, :k]  # topics have negative values!

# The NMF alternative (non-negative topics)
def nmf_topics(term_doc_matrix, k=10):
    A, W = separable_nmf(term_doc_matrix, k)  # A ≥ 0, W ≥ 0
    return A, W  # Interpretable, non-negative topic-word matrix
```

---

### Section 2.2: Algebraic Algorithms  
**Why is NMF computationally hard?**

**Key Concepts:**
- **rank(M)** vs **rank_+(M)** (nonnegative rank)
- Example: rank(M) = 3, rank_+(M) ≥ log(n) for M[i,j] = (i-j)²
- **NP-hardness**: General NMF is computationally intractable
- **Variable reduction**: 2r² variables instead of nr + mr for simplicial case

**Mathematical Insight:**
```
NMF Problem: Find A ≥ 0, W ≥ 0 such that M = AW
↓ (reformulate as polynomial system)
System: {M = AW, A ≥ 0, W ≥ 0}
↓ (variable reduction magic)  
Equivalent system with only 2r² variables!
```

**Why This Matters:** Even though NMF is NP-hard in general, understanding the algebraic structure points us toward tractable special cases.

---

### Section 2.3: Stability and Separability ⭐
**The breakthrough: Polynomial-time algorithm for separable matrices**

**Key Concepts:**
- **Geometric interpretation**: M = AW ⟺ C_M ⊆ C_A (cone containment)
- **Separability condition**: Each topic has an "anchor word"
- **Anchor word**: Word that appears almost exclusively in one topic
- **Polynomial algorithm**: Find anchors → Solve for A, W

**Interactive Demo:** [Separable NMF & Anchor Words](./separable_nmf_demo.html)
- Visualize separable vs non-separable matrices
- See the anchor words algorithm in action
- Understand convex hull geometry

**The Algorithm:**
```python
def anchor_words_algorithm(M):
    """
    Input: M = AW where A is separable, W has full row rank
    Output: A, W (up to rescaling)
    """
    # Step 1: Find anchor words (extreme points of convex hull)
    anchor_indices = find_anchors(M)
    
    # Step 2: Solve for A using anchor words
    A = solve_for_topics(M, anchor_indices)
    
    # Step 3: Solve for W  
    W = solve_linear_system(M, A)
    
    return A, W

def find_anchors(M):
    """Remove words that lie in convex hull of others"""
    I = set(range(len(M)))
    for i in range(len(M)):
        if M[i] in convex_hull(M[j] for j in I if j != i):
            I.discard(i)
    return list(I)  # Remaining words are anchors
```

**Real-world Examples of Anchor Words:**
- **Politics topic**: "401k", "filibuster", "senate" 
- **Sports topic**: "touchdown", "penalty", "referee"
- **Technology topic**: "algorithm", "database", "API"

---

### Section 2.4: Topic Models 🚀
**The complete pipeline: Text → Topics via separable NMF**

**Key Concepts:**
- **Generative model**: Documents as mixtures of topics
- **Gram matrix**: G = ARA^T where G[i,j] = P[word1=i, word2=j]  
- **Key insight**: G has same separable structure as A!
- **Recovery algorithm**: Gram matrix → Anchor words → Topics

**Interactive Demo:** [Topic Models & Recovery](./topic_models_demo.html)
- Generate synthetic documents from topic model
- Watch the complete recovery pipeline
- Compare recovered vs true topics

**Complete Pipeline:**
```python
def topic_model_recovery(documents, num_topics):
    """Complete pipeline from raw text to recovered topics"""
    
    # Step 1: Build term-document matrix
    M = build_term_document_matrix(documents)
    
    # Step 2: Compute Gram matrix from word co-occurrences  
    G = compute_gram_matrix(documents)
    
    # Step 3: Find anchor words in G (same structure as in A!)
    anchors = find_anchor_words(G)
    
    # Step 4: Solve linear system for P[topic|word]
    posterior_probs = solve_linear_system(G, anchors)
    
    # Step 5: Apply Bayes rule to get P[word|topic] = topic matrix
    A = apply_bayes_rule(posterior_probs, word_frequencies)
    
    return A, anchors
```

**Experimental Validation:**
- **Dataset**: 300K NY Times articles  
- **Performance**: 90% of topics had good anchor words
- **Speed**: Hundreds of times faster than MALLET
- **Quality**: Comparable or better topic interpretability

---

## 🛠️ Implementation Files

```
chapter2_nmf/
├── R/
│   ├── svd_utils.R                     # SVD analysis and visualization
│   ├── geometric_nmf.R                 # Convex hull & cone computations
│   ├── anchor_words.R                  # R implementation of algorithm
│   ├── topic_modeling.R                # Topic model simulation & recovery
│   └── visualization.R                 # ggplot2 visualizations
├── data/
│   ├── synthetic_documents.txt         # Generated document collection
│   ├── sample_news.txt                 # Real news article sample
│   └── topic_matrices/                 # Saved A, W matrices for testing
├── notebooks/
│   ├── 01_svd_vs_nmf_comparison.ipynb  # Why NMF beats SVD for topics
│   ├── 02_rank_vs_nonneg_rank.ipynb    # Algebraic complexity examples
│   ├── 03_separability_analysis.ipynb  # Geometric intuition
│   ├── 04_anchor_words_algorithm.ipynb # Step-by-step algorithm
│   └── 05_complete_topic_pipeline.ipynb # End-to-end topic modeling
├── exercises/
│   ├── problem_2_1_nonnegative_rank.py # Chapter exercises with solutions
│   ├── problem_2_2_rank_gap_example.py
│   ├── problem_2_3_papadimitriou_model.py
│   ├── problem_2_4_greedy_anchors.py
│   └── solutions/
└── tests/
    ├── test_anchor_algorithm.py        # Unit tests for core algorithms
    ├── test_topic_recovery.py          # End-to-end pipeline tests
    └── benchmarks/                     # Performance comparisons
```

---

## 🚀 Quick Start

### 1. Basic SVD vs NMF Comparison
```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from chapter2_nmf import separable_nmf, visualize_topics

# Load sample documents
documents = load_sample_documents()
M = build_term_document_matrix(documents)

# SVD approach (problems: negative values, orthogonal topics)
svd = TruncatedSVD(n_components=3)
svd_topics = svd.fit_transform(M.T).T  # Can have negative values!

# Our separable NMF approach  
A, W = separable_nmf(M, num_topics=3)  # A ≥ 0, interpretable!

# Compare topic quality
visualize_topics(svd_topics, method="SVD")
visualize_topics(A, method="Separable NMF")
```

### 2. Run the Anchor Words Algorithm
```python
from chapter2_nmf import AnchorWordsAlgorithm

# Create algorithm instance
alg = AnchorWordsAlgorithm(num_topics=3)

# Fit on your document collection
alg.fit(documents)

# Examine discovered anchor words
print("Discovered Anchor Words:")
for i, anchor in enumerate(alg.anchor_words):
    print(f"Topic {i+1}: {anchor} (confidence: {alg.anchor_scores[i]:.3f})")

# Get topic-word matrix
A = alg.get_topic_word_matrix()
print("\nTop words per topic:")
alg.print_top_words(n_words=10)
```

### 3. Complete Topic Modeling Pipeline
```python
from chapter2_nmf import TopicModelPipeline

# Initialize pipeline
pipeline = TopicModelPipeline(
    num_topics=5,
    min_anchor_confidence=0.7,
    use_gram_matrix=True
)

# Fit on document collection
results = pipeline.fit_transform(documents)

# Access results
topics = results['topic_word_matrix']  # A matrix
doc_topics = results['doc_topic_matrix']  # W matrix  
anchor_words = results['anchor_words']
quality_metrics = results['metrics']

print(f"Topic coherence: {quality_metrics['coherence']:.3f}")
print(f"Anchor word quality: {quality_metrics['anchor_confidence']:.3f}")
```

---

## 📊 Key Visualizations

### 1. SVD vs NMF Topic Comparison
Shows why negative values in SVD topics are uninterpretable:
```python
# SVD topic might look like:
# Topic 1: [politics: 0.5, sports: -0.3, tech: 0.8, ...]
# What does "negative sports contribution" mean?

# NMF topic looks like:
# Topic 1: [politics: 0.6, sports: 0.0, tech: 0.4, ...]  
# Clear: mostly politics, some tech, no sports
```

### 2. Convex Hull Geometry
Visualize why anchor words are extreme points:
```python
from chapter2_nmf import plot_convex_hull_demo

# Show word vectors in 2D topic space
plot_convex_hull_demo(
    word_vectors=M,  # rows of M are word vectors
    anchor_indices=[0, 5, 10],  # discovered anchors
    vocabulary=vocab_list
)
# Anchor words appear at vertices of convex hull!
```

### 3. Topic Recovery Quality
Track algorithm performance vs noise/separability:
```python
from chapter2_nmf import separability_analysis

# Test robustness to non-separability
results = separability_analysis(
    separability_levels=[0.9, 0.8, 0.7, 0.6, 0.5],
    noise_levels=[0.0, 0.1, 0.2, 0.3],
    num_trials=50
)

plot_recovery_quality(results)
# Shows graceful degradation as separability assumption weakens
```

---

## 🧠 Mathematical Deep Dives

### The Geometric Intuition
The key insight is that **NMF is fundamentally about cone geometry**:

```
Given: M = AW where A ≥ 0, W ≥ 0
Geometric meaning: Cone(M) ⊆ Cone(A)

In separable case:
- Word vectors lie in convex hull of topic vectors  
- Anchor words are vertices (extreme points)
- Algorithm finds vertices → recovers topics
```

### Why the Gram Matrix Works
The mathematical magic behind Section 2.4:

```
Gram Matrix: G[i,j] = P[word1=i, word2=j]

Key identity: G = A R A^T where:
- A = topic-word matrix (what we want)
- R = topic co-occurrence matrix  

Insight: G inherits separable structure from A!
→ Apply separable NMF to G
→ Recover A via Bayes rule
```

### Complexity Analysis
Why this matters computationally:

```
General NMF: NP-hard
   ↓ (add separability assumption)
Separable NMF: O(m³n) polynomial time
   ↓ (use greedy anchor finding)  
Practical algorithm: O(mn²) with high probability
```

---

## 🔬 Experimental Validation

### Synthetic Data Tests
```python
# Generate synthetic separable topics
true_A, true_W = generate_separable_topics(
    num_topics=5, 
    vocab_size=100,
    separability=0.8  # 80% separable
)

# Add realistic noise
noisy_M = true_A @ true_W + noise_matrix

# Test recovery quality
recovered_A, recovered_W = anchor_words_algorithm(noisy_M)
recovery_error = compute_recovery_error(true_A, recovered_A)

print(f"Recovery error: {recovery_error:.4f}")
# Should be small for separable case!
```

### Real Data Results
Replicating the book's NY Times experiments:
```python
# Load NY Times dataset (300K articles)
nyt_corpus = load_nyt_dataset()

# Apply our algorithm  
pipeline = TopicModelPipeline(num_topics=200)
results = pipeline.fit_transform(nyt_corpus)

# Measure separability in real data
separability_fraction = measure_separability(results['topics'])
print(f"Fraction of separable topics: {separability_fraction:.2f}")
# Paper reports ~0.9 - most real topics have anchor words!

# Compare speed vs MALLET
timing_comparison = benchmark_vs_mallet(nyt_corpus)
# Expected: 100x+ speedup with comparable quality
```

---

## 🎯 Exercises & Problems

### Problem 2-1: Nonnegative Rank Definitions
Explore equivalent definitions of nonnegative rank:
- Smallest r for sum of r nonnegative rank-1 matrices
- Cone generation perspective  
- Vertex characterization

**Solution approach:** Use geometric intuition and counterexamples.

### Problem 2-2: Rank Gap Example  
Prove that M[i,j] = (i-j)² has rank(M) = 3 but rank_+(M) ≥ log n.

**Key insight:** Zero pattern requires many nonnegative rectangles to cover.

### Problem 2-3: Papadimitriou Model
Analyze SVD behavior on block-diagonal document model.

**Result:** SVD provably recovers topics when support is disjoint.

### Problem 2-4: Greedy Anchor Algorithm
Prove correctness of greedy furthest-point anchor finding.

**Tool:** Strict convexity of ℓ2 norm ensures extreme points are furthest.

---

## 📚 Further Reading & Extensions

### Theoretical Extensions
- **Overcomplete case**: What if rank > min(m,n)?
- **Approximate separability**: Robustness analysis
- **Online algorithms**: Streaming anchor word detection

### Practical Applications  
- **Hyperspectral imaging**: Endmember detection
- **Collaborative filtering**: User-item matrix factorization  
- **Gene expression**: Pathway analysis
- **Audio processing**: Source separation

### Modern Connections
- **Deep learning**: Non-negative autoencoders
- **Optimization**: Interior point methods for NMF
- **High dimensions**: Random projection preprocessing

---

## 🏆 Chapter Summary

**What We Learned:**
1. **SVD limitations** for interpretable factorization (negative values, orthogonality)
2. **NMF hardness** in general (NP-complete, exponential algorithms)  
3. **Separability assumption** enables polynomial-time algorithms
4. **Anchor words method** works on real data (90% of topics have anchors)
5. **Complete pipeline** from text to interpretable topics

**Key Takeaway:** By moving beyond worst-case analysis and assuming realistic structure (separability), we can solve previously intractable problems with provable guarantees.

**Next Chapter Preview:** Tensor decompositions tackle the "rotation problem" in factor analysis - stay tuned for 3D generalizations of these ideas!

---

## 🔗 Links & Resources

- **Book PDF**: [Algorithmic Aspects of Machine Learning - Moitra](https://people.csail.mit.edu/moitra/docs/bookexv2.pdf)
- **Original Papers**: Arora et al. "A Practical Algorithm for Topic Modeling with Provable Guarantees" (ICML 2013)
- **Code Repository**: [GitHub - Separable NMF Implementation](https://github.com/your-repo/chapter2-nmf)
- **Interactive Demos**: Run locally or try online versions

---

*"The important thing is not to stop questioning. Curiosity has its own reason for existing."* - Einstein

The beauty of this chapter is seeing how theoretical computer science and machine learning inform each other - understanding why heuristics work leads to better algorithms! 🚀EADME.md                           # This guide
├── demos/
│   ├── svd_fundamentals_demo.html      # Section 2.1 interactive demo
│   ├── separable_nmf_demo.html         # Section 2.3 interactive demo  
│   └── topic_models_demo.html          # Section 2.4 complete pipeline
├── python/
│   ├── svd_utils.py                    # SVD review & Eckart-Young
│   ├── geometric_nmf.py                # Cone geometry & separability
│   ├── anchor_words.py                 # Core algorithm implementation
│   ├── topic_modeling.py               # Full topic model pipeline
│   └── visualization.py                # Plotting utilities
├── R