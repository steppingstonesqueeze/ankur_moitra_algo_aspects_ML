
# ============================================================================
# FILE: shared/sample_data.R
# ============================================================================

# Generate sample data for all apps
create_sample_data <- function() {
  # Sample matrix for SVD demo
  sample_matrix <- matrix(c(
    4, 2, 1, 0,
    2, 3, 1, 1,
    1, 1, 2, 1,
    0, 1, 1, 1
  ), nrow = 4, byrow = TRUE)
  
  # Document data for LSI demo  
  documents_lsi <- data.frame(
    Document = c("Sports Article", "Election News", "AI Research", "Olympic Games", "Tech Policy"),
    Politics = c(0.1, 0.9, 0.1, 0.2, 0.4),
    Sports = c(0.8, 0.05, 0.1, 0.7, 0.1),
    Technology = c(0.1, 0.05, 0.8, 0.1, 0.5),
    stringsAsFactors = FALSE
  )
  
  # Generate separable topic data
  topic_data <- generate_separable_topics(num_topics = 3, vocab_size = 15)
  
  list(
    sample_matrix = sample_matrix,
    documents_lsi = documents_lsi,
    separable_A = topic_data$A,
    vocabulary = topic_data$vocabulary,
    anchor_words = topic_data$anchor_words
  )
}

# Load sample data
SAMPLE_DATA <- create_sample_data()
