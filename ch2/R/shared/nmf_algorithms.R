
# ============================================================================
# FILE: shared/nmf_algorithms.R
# ============================================================================

# Shared algorithms for all apps
library(magrittr)

# Generate separable topic matrix
generate_separable_topics <- function(num_topics = 3, vocab_size = 15, separability = 0.8) {
  vocabulary <- c(
    "election", "vote", "candidate", "government", "policy",
    "game", "team", "player", "score", "match", 
    "tech", "software", "algorithm", "data", "computer"
  )[1:vocab_size]
  
  A <- matrix(0, nrow = vocab_size, ncol = num_topics)
  
  # Create separable structure
  words_per_topic <- vocab_size %/% num_topics
  
  for (i in 1:num_topics) {
    start_idx <- (i - 1) * words_per_topic + 1
    end_idx <- min(i * words_per_topic, vocab_size)
    
    # Anchor word gets high weight
    anchor_idx <- start_idx
    A[anchor_idx, i] <- 0.6
    
    # Topic words get moderate weights
    for (j in start_idx:end_idx) {
      if (j != anchor_idx) {
        A[j, i] <- runif(1, 0.1, 0.3)
      }
    }
    
    # Add some cross-topic noise if not perfectly separable
    if (separability < 1.0) {
      noise_indices <- setdiff(1:vocab_size, start_idx:end_idx)
      noise_count <- round(length(noise_indices) * (1 - separability))
      if (noise_count > 0) {
        noise_idx <- sample(noise_indices, noise_count)
        A[noise_idx, i] <- runif(noise_count, 0.01, 0.1)
      }
    }
  }
  
  # Normalize columns to sum to 1
  A <- apply(A, 2, function(x) x / sum(x))
  
  list(A = A, vocabulary = vocabulary, anchor_words = seq(1, vocab_size, by = words_per_topic)[1:num_topics])
}

# Generate documents from topic model
generate_documents <- function(A, num_docs = 10, doc_length = 20, noise_level = 0) {
  num_topics <- ncol(A)
  vocab_size <- nrow(A)
  
  documents <- list()
  doc_topic_mixtures <- matrix(0, nrow = num_docs, ncol = num_topics)
  
  for (d in 1:num_docs) {
    # Sample topic mixture
    mixture <- rgamma(num_topics, 2, 1)
    mixture <- mixture / sum(mixture)
    doc_topic_mixtures[d, ] <- mixture
    
    # Generate document
    doc_words <- c()
    for (w in 1:doc_length) {
      # Sample topic
      topic <- sample(1:num_topics, 1, prob = mixture)
      
      # Sample word from topic (with optional noise)
      word_probs <- A[, topic]
      if (noise_level > 0) {
        word_probs <- word_probs + runif(vocab_size, 0, noise_level)
        word_probs <- word_probs / sum(word_probs)
      }
      
      word <- sample(1:vocab_size, 1, prob = word_probs)
      doc_words <- c(doc_words, word)
    }
    
    documents[[d]] <- list(
      words = doc_words,
      topic_mixture = mixture,
      word_counts = table(factor(doc_words, levels = 1:vocab_size))
    )
  }
  
  list(documents = documents, doc_topic_mixtures = doc_topic_mixtures)
}

# Compute Gram matrix from documents
compute_gram_matrix <- function(documents, vocab_size) {
  G <- matrix(0, nrow = vocab_size, ncol = vocab_size)
  total_pairs <- 0
  
  for (doc in documents) {
    words <- doc$words
    if (length(words) > 1) {
      for (i in 1:(length(words) - 1)) {
        G[words[i], words[i + 1]] <- G[words[i], words[i + 1]] + 1
        total_pairs <- total_pairs + 1
      }
    }
  }
  
  # Normalize to probabilities
  if (total_pairs > 0) {
    G <- G / total_pairs
  }
  
  G
}

# Find anchor words (simplified greedy approach)
find_anchor_words <- function(G, num_topics = 3) {
  vocab_size <- nrow(G)
  anchor_candidates <- c()
  
  # Simple heuristic: words with high self-transition probability
  for (i in 1:vocab_size) {
    self_prob <- G[i, i]
    cross_prob <- mean(G[i, -i])
    score <- self_prob - cross_prob
    anchor_candidates <- c(anchor_candidates, score)
  }
  
  # Take top scoring words
  anchor_indices <- order(anchor_candidates, decreasing = TRUE)[1:min(num_topics, vocab_size)]
  
  list(indices = anchor_indices, scores = anchor_candidates[anchor_indices])
}

# SVD-related functions
compute_svd_approximation <- function(M, k) {
  svd_result <- svd(M)
  
  if (k > min(dim(M))) {
    k <- min(dim(M))
  }
  
  M_k <- svd_result$u[, 1:k, drop = FALSE] %*% 
    diag(svd_result$d[1:k], nrow = k) %*% 
    t(svd_result$v[, 1:k, drop = FALSE])
  
  error <- norm(M - M_k, "F")
  
  list(
    approximation = M_k,
    error = error,
    singular_values = svd_result$d,
    rank = k
  )
}
