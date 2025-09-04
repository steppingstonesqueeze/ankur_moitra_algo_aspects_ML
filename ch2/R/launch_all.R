# ============================================================================
# FILE: launch_all.R
# ============================================================================
launch_app <- function(app_name) {
  app_paths <- list(
    "svd" = "01_svd_fundamentals",
    "separable" = "02_separable_nmf", 
    "topics" = "03_topic_models"
  )
  
  if (!app_name %in% names(app_paths)) {
    cat("Available apps: svd, separable, topics\n")
    return(invisible(NULL))
  }
  
  app_path <- app_paths[[app_name]]
  
  if (!dir.exists(app_path)) {
    stop("App directory not found: ", app_path)
  }
  
  cat("🚀 Launching", app_name, "app...\n")
  shiny::runApp(app_path)
}

# Quick launch functions
launch_svd <- function() launch_app("svd")
launch_separable <- function() launch_app("separable") 
launch_topics <- function() launch_app("topics")

cat("📱 App launcher loaded! Use:\n")
cat("  launch_app('svd')        # SVD Fundamentals\n") 
cat("  launch_app('separable')  # Separable NMF\n")
cat("  launch_app('topics')     # Topic Models\n")
