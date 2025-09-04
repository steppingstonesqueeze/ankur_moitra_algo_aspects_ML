
# ============================================================================
# FILE: 01_svd_fundamentals/app.R
# ============================================================================

library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)
library(DT)
library(viridis)

# Source shared functions
source("../shared/nmf_algorithms.R", local = TRUE)
source("../shared/visualization_helpers.R", local = TRUE) 
source("../shared/sample_data.R", local = TRUE)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "SVD Fundamentals & NMF Motivation"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("SVD Basics", tabName = "basics", icon = icon("calculator")),
      menuItem("Rank-k Approximation", tabName = "approximation", icon = icon("chart-line")),
      menuItem("Latent Semantic Indexing", tabName = "lsi", icon = icon("search")),
      menuItem("Problems & NMF Solution", tabName = "problems", icon = icon("exclamation-triangle"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # SVD Basics Tab
      tabItem(tabName = "basics",
              fluidRow(
                box(width = 12, title = "Singular Value Decomposition: M = UΣV^T", status = "primary", solidHeader = TRUE,
                    h4("Key Properties:"),
                    tags$ul(
                      tags$li(tags$b("Existence:"), "Every matrix has an SVD"),
                      tags$li(tags$b("Algorithms:"), "Computable in O(mn²) time"),
                      tags$li(tags$b("Uniqueness:"), "Unique if singular values are distinct"), 
                      tags$li(tags$b("Best Approximation:"), "Optimal rank-k approximation")
                    ),
                    
                    h4("Frobenius Norm (Definition 2.1.1):"),
                    withMathJax("$$||M||_F = \\sqrt{\\sum_{i,j} M_{ij}^2} = \\sqrt{\\sum_i \\sigma_i^2}$$"),
                    p("Rotationally invariant norm - key for SVD optimality")
                )
              ),
              
              fluidRow(
                box(width = 6, title = "Sample Matrix (4×4)", status = "info", solidHeader = TRUE,
                    verbatimTextOutput("sample_matrix_output"),
                    h5("Properties:"),
                    verbatimTextOutput("matrix_properties")
                ),
                
                box(width = 6, title = "SVD Components", status = "info", solidHeader = TRUE,
                    verbatimTextOutput("svd_components")
                )
              )
      ),
      
      # Rank-k Approximation Tab  
      tabItem(tabName = "approximation",
              fluidRow(
                box(width = 12, title = "Eckart-Young Theorem", status = "success", solidHeader = TRUE,
                    withMathJax("Best rank-k approximation: $$M_k = \\sum_{i=1}^k \\sigma_i u_i v_i^T$$"),
                    
                    h4("Interactive Controls:"),
                    sliderInput("rank_k", "Approximation Rank:", min = 1, max = 4, value = 2, step = 1)
                )
              ),
              
              fluidRow(
                box(width = 8, title = "Approximation Quality vs Rank", status = "primary", solidHeader = TRUE,
                    plotOutput("approximation_plot", height = "400px")
                ),
                
                box(width = 4, title = "Error Analysis", status = "info", solidHeader = TRUE,
                    h5("Current Analysis:"),
                    verbatimTextOutput("error_analysis"),
                    br(),
                    h6("Formula:"),
                    withMathJax("$$||M - M_k||_F = \\sqrt{\\sum_{i=k+1}^r \\sigma_i^2}$$")
                )
              )
      ),
      
      # LSI Tab
      tabItem(tabName = "lsi", 
              fluidRow(
                box(width = 12, title = "Latent Semantic Indexing (LSI)", status = "warning", solidHeader = TRUE,
                    p("Document similarity using SVD on term-by-document matrix"),
                    withMathJax("$$\\text{similarity}(\\text{doc}_i, \\text{doc}_j) = \\langle U_{1...k}^T M_i, U_{1...k}^T M_j \\rangle$$"),
                    p("Project documents into k-dimensional 'topic space' using top k singular vectors")
                )
              ),
              
              fluidRow(
                box(width = 6, title = "Document Compositions", status = "info", solidHeader = TRUE,
                    DT::dataTableOutput("document_table")
                ),
                
                box(width = 6, title = "Topic Distribution", status = "primary", solidHeader = TRUE,
                    plotOutput("lsi_plot", height = "350px")
                )
              )
      ),
      
      # Problems Tab
      tabItem(tabName = "problems",
              fluidRow(
                box(width = 12, title = "Problems with SVD for Text Analysis", status = "danger", solidHeader = TRUE,
                    h4("Issue 1: Orthogonal Topics", style = "color: #d32f2f;"),
                    p("SVD forces topics to be orthogonal, but real topics overlap!"),
                    div(style = "background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;",
                        'Topic "Politics": [election, vote, candidate, economy, ...]', br(),
                        'Topic "Finance": [stock, economy, market, trade, ...]', br(),
                        span('⟨politics, finance⟩ = 0 (forced by SVD)', style = "color: #d32f2f; font-weight: bold;")
                    ),
                    
                    br(),
                    h4("Issue 2: Negative Values", style = "color: #d32f2f;"),
                    p("Topics can have negative word weights - no interpretation!"),
                    div(style = "background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;",
                        'Topic vector: [0.5, ', span('-0.3', style = "color: #d32f2f; font-weight: bold;"), ', 0.8, ', 
                        span('-0.1', style = "color: #d32f2f; font-weight: bold;"), ', ...]', br(),
                        span('What does "negative contribution" mean?', style = "color: #d32f2f;")
                    )
                )
              ),
              
              fluidRow(
                box(width = 12, title = "NMF Solution (Definition 2.1.7)", status = "success", solidHeader = TRUE,
                    withMathJax("$$M = AW \\text{ where } A \\geq 0, W \\geq 0$$"),
                    
                    fluidRow(
                      column(6, 
                             h5("Interpretation:"),
                             tags$ul(
                               tags$li(tags$b("A:"), "Topic-word matrix (non-negative)"),
                               tags$li(tags$b("W:"), "Document-topic matrix (non-negative)"),
                               tags$li("Documents are convex combinations of topics")
                             )
                      ),
                      column(6,
                             h5("Challenge:"), 
                             p(tags$b("Problem:"), "NP-hard in general!", style = "color: #d32f2f;"),
                             p(tags$b("Solution:"), "Find special cases (separability)")
                      )
                    )
                )
              ),
              
              fluidRow(
                box(width = 12, title = "SVD vs NMF Comparison", status = "info", solidHeader = TRUE,
                    DT::dataTableOutput("comparison_table")
                )
              ),
              
              fluidRow(
                box(width = 12, title = "Chapter 2 Roadmap", status = "primary", solidHeader = TRUE,
                    div(
                      p(tags$b("Problem:"), "NMF is NP-hard in general"),
                      p(tags$b("Solution:"), "Find special cases with polynomial algorithms"),  
                      p(tags$b("Key Insight:"), '"Separability" assumption → Anchor Words Algorithm')
                    )
                )
              )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Sample matrix output
  output$sample_matrix_output <- renderText({
    paste(capture.output(print(SAMPLE_DATA$sample_matrix)), collapse = "\n")
  })
  
  # Matrix properties
  output$matrix_properties <- renderText({
    M <- SAMPLE_DATA$sample_matrix
    frobenius_norm <- norm(M, "F")
    paste0("Frobenius Norm: ", round(frobenius_norm, 2))
  })
  
  # SVD components
  output$svd_components <- renderText({
    svd_result <- svd(SAMPLE_DATA$sample_matrix)
    paste0("Singular Values: ", paste(round(svd_result$d, 2), collapse = ", "), "\n",
           "Rank: ", sum(svd_result$d > 1e-10))
  })
  
  # Approximation plot
  output$approximation_plot <- renderPlot({
    M <- SAMPLE_DATA$sample_matrix
    svd_result <- svd(M)
    plot_approximation_quality(svd_result$d, max_rank = 4)
  })
  
  # Error analysis
  output$error_analysis <- renderText({
    M <- SAMPLE_DATA$sample_matrix
    k <- input$rank_k
    approx_result <- compute_svd_approximation(M, k)
    
    retained_values <- paste(round(approx_result$singular_values[1:k], 2), collapse = ", ")
    
    paste0("Current rank: ", k, "\n",
           "Frobenius error: ", round(approx_result$error, 3), "\n",
           "Retained σᵢ: ", retained_values)
  })
  
  # Document table for LSI
  output$document_table <- DT::renderDataTable({
    SAMPLE_DATA$documents_lsi
  }, options = list(pageLength = 10, scrollX = TRUE))
  
  # LSI plot
  output$lsi_plot <- renderPlot({
    plot_document_topics(as.matrix(SAMPLE_DATA$documents_lsi[, -1]), 
                         doc_names = SAMPLE_DATA$documents_lsi$Document)
  })
  
  # Comparison table
  output$comparison_table <- DT::renderDataTable({
    create_comparison_table()
  }, options = list(pageLength = 10, dom = 't')) 
}

# Run app
shinyApp(ui = ui, server = server)