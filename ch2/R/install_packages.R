# ============================================================================
# FILE: install_packages.R  
# ============================================================================
install_packages <- function() {
  cat("Installing required packages for Chapter 2 NMF Shiny Apps...\n")
  
  required_packages <- c(
    "shiny",           # Core Shiny framework
    "shinydashboard",  # Dashboard layout
    "ggplot2",         # Grammar of graphics
    "plotly",          # Interactive plots
    "DT",              # Interactive tables
    "viridis",         # Color palettes
    "gridExtra",       # Grid layouts
    "reshape2",        # Data reshaping
    "dplyr",           # Data manipulation
    "magrittr"         # Pipe operator
  )
  
  new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
  
  if(length(new_packages)) {
    cat("Installing:", paste(new_packages, collapse=", "), "\n")
    install.packages(new_packages, dependencies = TRUE)
  } else {
    cat("All required packages are already installed!\n")
  }
  
  cat("✅ Installation complete!\n")
}

# Run installation
install_packages()