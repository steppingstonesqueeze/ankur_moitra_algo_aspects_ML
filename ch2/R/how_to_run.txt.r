# ============================================================================
# CHAPTER 2 NMF - R SHINY COMPLETE PACKAGE
# ============================================================================

# FILE STRUCTURE:
# chapter2_nmf_shiny/
# ├── README.md
# ├── install_packages.R
# ├── launch_all.R
# ├── shared/
# │   ├── nmf_algorithms.R
# │   ├── visualization_helpers.R
# │   └── sample_data.R
# ├── 01_svd_fundamentals/
# │   └── app.R
# ├── 02_separable_nmf/
# │   └── app.R
# └── 03_topic_models/
#     └── app.R

# ============================================================================
# FILE: README.md
# ============================================================================
# # Chapter 2: Nonnegative Matrix Factorization - R Shiny Apps
# 
# ## 🚀 Quick Start
# 
# 1. **Install packages:**
#    ```r
#    source("install_packages.R")
#    ```
# 
# 2. **Launch apps:**
#    ```r
#    source("launch_all.R")
#    launch_app("svd")        # SVD Fundamentals
#    launch_app("separable")  # Separable NMF
#    launch_app("topics")     # Topic Models
#    ```
# 
# ## 📱 Individual Apps
# 
# ### 1. SVD Fundamentals (`01_svd_fundamentals/`)
# - SVD vs NMF comparison
# - Rank-k approximation interactive demo  
# - Latent Semantic Indexing visualization
# - Problems with SVD for text analysis
# 
# ### 2. Separable NMF (`02_separable_nmf/`)
# - Geometric interpretation of NMF
# - Separability condition demonstration
# - Anchor Words Algorithm walkthrough
# - Separable vs non-separable comparison
# 
# ### 3. Topic Models (`03_topic_models/`)
# - Complete generative model simulation
# - Gram matrix method visualization
# - Topic recovery pipeline
# - Performance evaluation and results
# 
# ## 📚 Educational Flow
# 
# Work through the apps in order:
# 1. **SVD Fundamentals** - Understand why NMF is needed
# 2. **Separable NMF** - Learn the polynomial-time breakthrough  
# 3. **Topic Models** - See the complete real-world application
# 
# ## 🛠️ Technical Requirements
# 
# - R >= 4.0.0
# - Packages: shiny, ggplot2, plotly, DT, shinydashboard, viridis
# - Modern web browser
# 
# ## 📖 Based on:
# "Algorithmic Aspects of Machine Learning" by Ankur Moitra (MIT)
# Chapter 2: Nonnegative Matrix Factorization



# ============================================================================
# END OF R SHINY PACKAGE
# ============================================================================

# USAGE INSTRUCTIONS:
# 
# 1. Save all files in the structure shown above
# 2. Run: source("install_packages.R")
# 3. Run: source("launch_all.R") 
# 4. Launch apps:
#    - launch_app("svd")        # SVD Fundamentals
#    - launch_app("separable")  # Separable NMF  
#    - launch_app("topics")     # Topic Models
#
# Each app is a complete interactive demonstration of the corresponding 
# section from Chapter 2 of "Algorithmic Aspects of Machine Learning"