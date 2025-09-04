
# ============================================================================
# FILE: shared/visualization_helpers.R
# ============================================================================

library(ggplot2)
library(plotly)
library(viridis)

# Custom theme for all plots
nmf_theme <- function() {
  theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      panel.grid.minor = element_blank()
    )
}

# Plot matrix as heatmap
plot_matrix_heatmap <- function(mat, title = "Matrix", show_values = TRUE) {
  # Convert matrix to long format
  mat_long <- expand.grid(Row = 1:nrow(mat), Col = 1:ncol(mat))
  mat_long$Value <- as.vector(mat)
  
  p <- ggplot(mat_long, aes(x = Col, y = Row, fill = Value)) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_viridis_c(name = "Value") +
    scale_y_reverse() +
    labs(title = title, x = "Column", y = "Row") +
    nmf_theme() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (show_values && prod(dim(mat)) <= 100) {
    p <- p + geom_text(aes(label = round(Value, 2)), color = "white", size = 3)
  }
  
  p
}

# Plot SVD approximation quality
plot_approximation_quality <- function(singular_values, max_rank = NULL) {
  if (is.null(max_rank)) {
    max_rank <- length(singular_values)
  }
  
  ranks <- 1:min(max_rank, length(singular_values))
  errors <- sapply(ranks, function(k) {
    sqrt(sum(singular_values[(k+1):length(singular_values)]^2))
  })
  
  df <- data.frame(Rank = ranks, Error = errors)
  
  ggplot(df, aes(x = Rank, y = Error)) +
    geom_line(color = "#2563eb", size = 1.2) +
    geom_point(color = "#2563eb", size = 3) +
    labs(title = "Approximation Error vs Rank",
         x = "Rank k", 
         y = "Frobenius Error") +
    nmf_theme()
}

# Plot document topic distributions
plot_document_topics <- function(doc_topic_mixtures, doc_names = NULL) {
  if (is.null(doc_names)) {
    doc_names <- paste("Doc", 1:nrow(doc_topic_mixtures))
  }
  
  # Convert to long format
  df <- data.frame(
    Document = rep(doc_names, ncol(doc_topic_mixtures)),
    Topic = rep(paste("Topic", 1:ncol(doc_topic_mixtures)), each = nrow(doc_topic_mixtures)),
    Proportion = as.vector(doc_topic_mixtures)
  )
  
  ggplot(df, aes(x = Document, y = Proportion, fill = Topic)) +
    geom_bar(stat = "identity") +
    scale_fill_viridis_d() +
    labs(title = "Document Topic Distributions",
         x = "Documents", 
         y = "Topic Proportion") +
    nmf_theme() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot separability analysis
plot_separability_scatter <- function(A, vocabulary, anchor_indices = NULL) {
  # Use first two topics for 2D plot
  if (ncol(A) < 2) {
    stop("Need at least 2 topics for scatter plot")
  }
  
  df <- data.frame(
    Topic1 = A[, 1],
    Topic2 = A[, 2], 
    Word = vocabulary,
    IsAnchor = 1:nrow(A) %in% anchor_indices
  )
  
  ggplot(df, aes(x = Topic1, y = Topic2, color = IsAnchor, size = IsAnchor)) +
    geom_point(alpha = 0.7) +
    scale_color_manual(values = c("FALSE" = "#3b82f6", "TRUE" = "#ef4444"),
                       labels = c("Regular Words", "Anchor Words")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 5), guide = "none") +
    geom_text(aes(label = Word), vjust = -0.8, size = 3, show.legend = FALSE) +
    labs(title = "Words in Topic Space",
         x = "Topic 1 Weight",
         y = "Topic 2 Weight",
         color = "Word Type") +
    nmf_theme()
}

# Create comparison table
create_comparison_table <- function() {
  data.frame(
    Feature = c("Orthogonal Topics", "Non-negative Values", "Computational Complexity", "Interpretability", "Uniqueness"),
    SVD = c("Yes", "No", "O(mn²)", "Low", "Unique"),
    NMF = c("No", "Yes", "NP-hard", "High", "Non-unique"),
    stringsAsFactors = FALSE
  )
}
