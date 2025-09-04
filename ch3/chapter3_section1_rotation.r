# ============================================================================
# CHAPTER 3.1: THE ROTATION PROBLEM - R SHINY APP
# ============================================================================
# 
# This app demonstrates why matrix factorizations have rotational ambiguity
# and motivates the need for tensor methods in factor analysis.
#
# Based on: "Algorithmic Aspects of Machine Learning" by Ankur Moitra
# Chapter 3.1: The Rotation Problem (Pages 46-47)
#
# SAVE AS: 04_rotation_problem/app.R
# ============================================================================

library(shiny)
library(shinydashboard)
library(ggplot2)
library(plotly)
library(DT)
library(viridis)
library(reshape2)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Generate Spearman's student test data
generate_student_data <- function(n_students = 100, noise_level = 0.1) {
  # True factors: mathematical and verbal intelligence
  math_ability <- rnorm(n_students, mean = 100, sd = 15)
  verbal_ability <- rnorm(n_students, mean = 100, sd = 15)
  
  # Test loadings: how much each test depends on each factor
  # Rows: tests, Columns: factors (math, verbal)
  test_loadings <- matrix(c(
    # Math tests
    0.8, 0.2,  # Algebra
    0.9, 0.1,  # Geometry  
    0.7, 0.3,  # Calculus
    0.6, 0.4,  # Statistics
    0.8, 0.2,  # Logic
    # Verbal tests
    0.2, 0.8,  # Reading
    0.1, 0.9,  # Writing
    0.3, 0.7,  # Vocabulary
    0.4, 0.6,  # Grammar
    0.2, 0.8   # Literature
  ), nrow = 10, ncol = 2, byrow = TRUE)
  
  # Test names
  test_names <- c("Algebra", "Geometry", "Calculus", "Statistics", "Logic",
                  "Reading", "Writing", "Vocabulary", "Grammar", "Literature")
  
  # Student abilities matrix (n_students x 2)
  A <- cbind(math_ability, verbal_ability)
  
  # Generate test scores: M = A %*% t(B) + noise
  B <- test_loadings  # 10 tests x 2 factors
  M <- A %*% t(B) + matrix(rnorm(n_students * 10, 0, noise_level * 50), 
                           nrow = n_students, ncol = 10)
  
  # Normalize scores to 0-100 range
  M <- pmax(0, pmin(100, M))
  
  list(
    scores = M,  # n_students x 10 test scores
    true_abilities = A,  # n_students x 2 true factors
    true_loadings = B,   # 10 x 2 true test loadings
    test_names = test_names,
    student_ids = paste("Student", 1:n_students)
  )
}

# Apply rotation to factors
apply_rotation <- function(A, B, theta) {
  # Rotation matrix
  rotation_matrix <- matrix(c(cos(theta), -sin(theta),
                             sin(theta), cos(theta)), 
                           nrow = 2, ncol = 2)
  
  # Rotated factors: A_rot = A %*% R, B_rot = B %*% R  
  A_rotated <- A %*% rotation_matrix
  B_rotated <- B %*% rotation_matrix
  
  list(
    A_rotated = A_rotated,
    B_rotated = B_rotated,
    rotation_matrix = rotation_matrix,
    theta = theta
  )
}

# Perform SVD factorization  
perform_svd_factorization <- function(M, rank = 2) {
  svd_result <- svd(M)
  
  # Take first 'rank' components
  U <- svd_result$u[, 1:rank, drop = FALSE] %*% diag(sqrt(svd_result$d[1:rank]), nrow = rank)
  V <- svd_result$v[, 1:rank, drop = FALSE] %*% diag(sqrt(svd_result$d[1:rank]), nrow = rank)
  
  # Reconstruct: M ≈ U %*% t(V)
  M_reconstructed <- U %*% t(V)
  reconstruction_error <- norm(M - M_reconstructed, "F")
  
  list(
    U = U,  # Student factors (n_students x 2)
    V = V,  # Test factors (n_tests x 2)
    reconstruction_error = reconstruction_error,
    singular_values = svd_result$d[1:rank]
  )
}

# ============================================================================
# SHINY UI
# ============================================================================

ui <- dashboardPage(
  dashboardHeader(title = "Chapter 3.1: The Rotation Problem"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Factor Analysis Setup", tabName = "setup", icon = icon("cogs")),
      menuItem("The Rotation Problem", tabName = "rotation", icon = icon("sync-alt")),
      menuItem("SVD vs Ground Truth", tabName = "comparison", icon = icon("balance-scale")),
      menuItem("Why Tensors Help", tabName = "solution", icon = icon("cube"))
    ),
    
    # Controls
    h4("Simulation Controls", style = "color: white; padding-left: 15px;"),
    div(style = "padding: 0 15px;",
        sliderInput("n_students", "Number of Students:", 
                   min = 50, max = 200, value = 100, step = 10),
        sliderInput("noise_level", "Noise Level:", 
                   min = 0, max = 0.3, value = 0.1, step = 0.05),
        actionButton("regenerate", "Regenerate Data", class = "btn-primary",
                    style = "margin-top: 10px;")
    )
  ),
  
  dashboardBody(
    # Custom CSS
    tags$head(tags$style(HTML("
      .rotation-box {
        background-color: #f0f9ff;
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
      }
      .problem-highlight {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
      }
      .solution-highlight {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
      }
    "))),
    
    tabItems(
      # Setup Tab
      tabItem(tabName = "setup",
        fluidRow(
          box(width = 12, title = "Spearman's Intelligence Theory", status = "primary", solidHeader = TRUE,
            div(class = "rotation-box",
                h4("The Historical Context"),
                p("Charles Spearman believed intelligence had two components:"),
                tags$ul(
                  tags$li(tags$b("Mathematical Intelligence:"), "Quantitative reasoning, logic, problem-solving"),
                  tags$li(tags$b("Verbal Intelligence:"), "Language skills, reading comprehension, vocabulary")
                ),
                p("He designed 10 different tests and measured 1000 students' performance to test this theory.")
            ),
            
            h4("The Factor Analysis Model:"),
            withMathJax("$$M = AB^T$$"),
            tags$ul(
              tags$li(tags$b("M:"), "Student scores matrix (1000 × 10)"),
              tags$li(tags$b("A:"), "Student abilities (1000 × 2) - what we want to find"),
              tags$li(tags$b("B:"), "Test loadings (10 × 2) - how tests relate to abilities")
            )
          )
        ),
        
        fluidRow(
          box(width = 6, title = "Generated Student Data", status = "info", solidHeader = TRUE,
            DT::dataTableOutput("student_scores_table"),
            p("Sample of student test scores (0-100 scale)", style = "font-size: 0.8em; color: #666;")
          ),
          
          box(width = 6, title = "True Factors (Ground Truth)", status = "success", solidHeader = TRUE,
            h5("Student Abilities (A matrix):"),
            plotOutput("true_abilities_plot", height = "200px"),
            br(),
            h5("Test Loadings (B matrix):"),
            DT::dataTableOutput("true_loadings_table")
          )
        )
      ),
      
      # Rotation Problem Tab  
      tabItem(tabName = "rotation",
        fluidRow(
          box(width = 12, title = "The Rotation Problem", status = "danger", solidHeader = TRUE,
            div(class = "problem-highlight",
                h4("🔄 The Core Issue: Non-Unique Factorizations"),
                withMathJax("$$M = AB^T = (AO)(O^T B^T) = \\tilde{A}\\tilde{B}^T$$"),
                p("For any orthogonal matrix O, we get a different factorization that fits the data equally well!"),
                p("But ", tags$code("Ã"), " and ", tags$code("B̃"), " have completely different interpretations.")
            )
          )
        ),
        
        fluidRow(
          box(width = 12, title = "Interactive Rotation Demo", status = "primary", solidHeader = TRUE,
            div(style = "text-align: center;",
                h4("Rotation Angle: ", textOutput("rotation_angle_display", inline = TRUE)),
                sliderInput("rotation_angle", "", 
                           min = 0, max = 360, value = 0, step = 5,
                           animate = animationOptions(interval = 200, loop = TRUE))
            )
          )
        ),
        
        fluidRow(
          box(width = 6, title = "Original Factors (A, B)", status = "info", solidHeader = TRUE,
            h5("Student Abilities Distribution:"),
            plotOutput("original_factors_plot", height = "250px"),
            h5("Interpretation:"),
            tags$ul(
              tags$li("X-axis: Mathematical ability"),
              tags$li("Y-axis: Verbal ability"),
              tags$li("Clear separation and meaning")
            )
          ),
          
          box(width = 6, title = "Rotated Factors (ÃO, B̃O)", status = "warning", solidHeader = TRUE,
            h5("Rotated Abilities Distribution:"),
            plotOutput("rotated_factors_plot", height = "250px"),
            h5("Problem:"),
            tags$ul(
              tags$li("Same data fit: ||M - ÃB̃^T||²"),
              tags$li("Different interpretation!"),
              tags$li("Which factors are 'correct'?")
            )
          )
        ),
        
        fluidRow(
          box(width = 12, title = "Reconstruction Quality", status = "success", solidHeader = TRUE,
            h4("Key Insight: Both factorizations are equally valid!"),
            div(id = "reconstruction_quality",
                style = "font-family: monospace; background-color: #f8f9fa; padding: 15px; border-radius: 5px;",
                "Original: ||M - AB^T||_F = ", textOutput("original_error", inline = TRUE), br(),
                "Rotated: ||M - ÃB̃^T||_F = ", textOutput("rotated_error", inline = TRUE), br(),
                "Difference: ", textOutput("error_difference", inline = TRUE)
            )
          )
        )
      ),
      
      # Comparison Tab
      tabItem(tabName = "comparison",
        fluidRow(
          box(width = 12, title = "SVD vs Ground Truth Comparison", status = "primary", solidHeader = TRUE,
            p("What happens when we apply SVD to recover factors? Does it find the 'true' factors?"),
            actionButton("run_svd", "Run SVD Factorization", class = "btn-primary")
          )
        ),
        
        fluidRow(
          box(width = 6, title = "Ground Truth Factors", status = "success", solidHeader = TRUE,
            plotOutput("ground_truth_comparison_plot", height = "300px"),
            h6("True mathematical vs verbal abilities")
          ),
          
          box(width = 6, title = "SVD Recovered Factors", status = "info", solidHeader = TRUE,
            plotOutput("svd_recovered_plot", height = "300px"),
            h6("SVD-derived factors (could be any rotation!)")
          )
        ),
        
        fluidRow(
          box(width = 12, title = "The Fundamental Problem", status = "danger", solidHeader = TRUE,
            div(class = "problem-highlight",
                h4("🎯 SVD Gives AN Answer, Not THE Answer"),
                fluidRow(
                  column(6,
                         h5("What SVD guarantees:"),
                         tags$ul(
                           tags$li("Optimal reconstruction: min ||M - UV^T||²"),
                           tags$li("Orthogonal factors: U^T U = I"),
                           tags$li("Unique decomposition (up to signs)")
                         )
                  ),
                  column(6,
                         h5("What SVD doesn't guarantee:"),
                         tags$ul(
                           tags$li(tags$b("Interpretability:"), "Factors may not match reality"),
                           tags$li(tags$b("Meaning:"), "No guarantee factors = math/verbal ability"),
                           tags$li(tags$b("Uniqueness:"), "Infinite rotational variants exist")
                         )
                  )
                )
            )
          )
        )
      ),
      
      # Solution Tab
      tabItem(tabName = "solution",
        fluidRow(
          box(width = 12, title = "Why Tensors Solve the Rotation Problem", status = "success", solidHeader = TRUE,
            div(class = "solution-highlight",
                h4("🎯 The Tensor Solution"),
                p("Tensors provide ", tags$b("multiple views"), " of the same factors, breaking the rotational symmetry!")
            )
          )
        ),
        
        fluidRow(
          box(width = 6, title = "Matrix Problem", status = "danger", solidHeader = TRUE,
            h5("Single View → Ambiguity"),
            withMathJax("$$M = AB^T \\text{ (one constraint)}$$"),
            div(style = "background-color: #fef2f2; padding: 15px; border-radius: 5px;",
                tags$ul(
                  tags$li("Only one matrix equation"),
                  tags$li("Infinite rotational solutions"),
                  tags$li("M = (AO)(O^T B^T) for any orthogonal O"),
                  tags$li("No way to prefer one over another")
                )
            )
          ),
          
          box(width = 6, title = "Tensor Solution", status = "success", solidHeader = TRUE,
            h5("Multiple Views → Uniqueness"),
            withMathJax("$$T = \\sum_{i=1}^r u^{(i)} \\otimes v^{(i)} \\otimes w^{(i)}$$"),
            div(style = "background-color: #f0fdf4; padding: 15px; border-radius: 5px;",
                tags$ul(
                  tags$li("Multiple tensor slices provide constraints"),
                  tags$li("Different 'views' of the same factors"),
                  tags$li("Rotation breaks consistency across views"),
                  tags$li("Unique decomposition under mild conditions!")
                )
            )
          )
        ),
        
        fluidRow(
          box(width = 12, title = "Intuitive Explanation", status = "info", solidHeader = TRUE,
            h4("🔍 Why Multiple Views Help"),
            fluidRow(
              column(4,
                     div(style = "text-align: center; padding: 20px;",
                         h5("View 1: T[:,:,1]"),
                         div(style = "background-color: #e3f2fd; padding: 10px; border-radius: 5px;",
                             "Matrix slice 1", br(),
                             "Shows factors in one context"
                         )
                     )
              ),
              column(4,
                     div(style = "text-align: center; padding: 20px;",
                         h5("View 2: T[:,:,2]"),
                         div(style = "background-color: #f3e5f5; padding: 10px; border-radius: 5px;",
                             "Matrix slice 2", br(),
                             "Shows factors in another context"
                         )
                     )
              ),
              column(4,
                     div(style = "text-align: center; padding: 20px;",
                         h5("View 3: T[:,:,3]"),
                         div(style = "background-color: #e8f5e8; padding: 10px; border-radius: 5px;",
                             "Matrix slice 3", br(),
                             "Shows factors in third context"
                         )
                     )
              )
            ),
            
            div(style = "text-align: center; margin-top: 20px;",
                h5("🎯 Key Insight:"),
                p("A rotation that preserves one view will typically ", tags$b("break"), " the other views!"),
                p("This constraint forces a ", tags$b("unique solution"), " (up to permutations and scaling).")
            )
          )
        ),
        
        fluidRow(
          box(width = 12, title = "Chapter 3 Preview", status = "primary", solidHeader = TRUE,
            h4("What's Coming Next:"),
            div(
              h5("🧊 Section 3.2: Tensor Fundamentals"),
              p("Deep dive into tensor operations, rank, and pathologies"),
              
              h5("⚙️ Section 3.3: Jennrich's Algorithm"), 
              p("The actual polynomial-time algorithm for tensor decomposition!"),
              
              h5("📊 Section 3.4: Perturbation Bounds"),
              p("Robustness analysis and noise handling")
            )
          )
        )
      )
    )
  )
)

# ============================================================================
# SHINY SERVER
# ============================================================================

server <- function(input, output, session) {
  
  # Reactive values
  values <- reactiveValues(
    data = NULL,
    svd_result = NULL
  )
  
  # Generate initial data
  observe({
    values$data <- generate_student_data(n_students = input$n_students, 
                                        noise_level = input$noise_level)
  })
  
  # Regenerate data when button clicked
  observeEvent(input$regenerate, {
    values$data <- generate_student_data(n_students = input$n_students, 
                                        noise_level = input$noise_level)
  })
  
  # Student scores table
  output$student_scores_table <- DT::renderDataTable({
    req(values$data)
    
    # Show first 10 students and first 5 tests
    display_data <- values$data$scores[1:min(10, nrow(values$data$scores)), 1:5]
    colnames(display_data) <- values$data$test_names[1:5]
    rownames(display_data) <- values$data$student_ids[1:min(10, nrow(values$data$scores))]
    
    round(display_data, 1)
  }, options = list(pageLength = 10, scrollX = TRUE))
  
  # True abilities plot
  output$true_abilities_plot <- renderPlot({
    req(values$data)
    
    df <- data.frame(
      Math = values$data$true_abilities[, 1],
      Verbal = values$data$true_abilities[, 2]
    )
    
    ggplot(df, aes(x = Math, y = Verbal)) +
      geom_point(alpha = 0.6, color = "#2563eb") +
      labs(title = "True Student Abilities",
           x = "Mathematical Intelligence", 
           y = "Verbal Intelligence") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  })
  
  # True loadings table
  output$true_loadings_table <- DT::renderDataTable({
    req(values$data)
    
    loadings_df <- data.frame(
      Test = values$data$test_names,
      Math_Loading = round(values$data$true_loadings[, 1], 2),
      Verbal_Loading = round(values$data$true_loadings[, 2], 2)
    )
    
    loadings_df
  }, options = list(pageLength = 10))
  
  # Rotation angle display
  output$rotation_angle_display <- renderText({
    paste0(input$rotation_angle, "°")
  })
  
  # Original factors plot
  output$original_factors_plot <- renderPlot({
    req(values$data)
    
    df <- data.frame(
      Math = values$data$true_abilities[, 1],
      Verbal = values$data$true_abilities[, 2]
    )
    
    ggplot(df, aes(x = Math, y = Verbal)) +
      geom_point(alpha = 0.6, color = "#2563eb", size = 2) +
      labs(title = "Original Factors",
           x = "Mathematical Ability", 
           y = "Verbal Ability") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
      coord_equal()
  })
  
  # Rotated factors plot
  output$rotated_factors_plot <- renderPlot({
    req(values$data)
    
    theta <- input$rotation_angle * pi / 180  # Convert to radians
    rotated <- apply_rotation(values$data$true_abilities, values$data$true_loadings, theta)
    
    df <- data.frame(
      Factor1 = rotated$A_rotated[, 1],
      Factor2 = rotated$A_rotated[, 2]
    )
    
    ggplot(df, aes(x = Factor1, y = Factor2)) +
      geom_point(alpha = 0.6, color = "#dc2626", size = 2) +
      labs(title = paste0("Rotated Factors (", input$rotation_angle, "°)"),
           x = "Rotated Factor 1", 
           y = "Rotated Factor 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
      coord_equal()
  })
  
  # Reconstruction errors
  output$original_error <- renderText({
    req(values$data)
    
    M <- values$data$scores
    A <- values$data$true_abilities
    B <- values$data$true_loadings
    
    M_reconstructed <- A %*% t(B)
    error <- norm(M - M_reconstructed, "F")
    round(error, 2)
  })
  
  output$rotated_error <- renderText({
    req(values$data)
    
    theta <- input$rotation_angle * pi / 180
    rotated <- apply_rotation(values$data$true_abilities, values$data$true_loadings, theta)
    
    M <- values$data$scores
    M_reconstructed <- rotated$A_rotated %*% t(rotated$B_rotated)
    error <- norm(M - M_reconstructed, "F")
    round(error, 2)
  })
  
  output$error_difference <- renderText({
    req(values$data)
    
    # Both should be essentially the same
    paste("≈ 0 (both factorizations are equivalent!)")
  })
  
  # SVD analysis
  observeEvent(input$run_svd, {
    req(values$data)
    values$svd_result <- perform_svd_factorization(values$data$scores, rank = 2)
  })
  
  # Ground truth comparison plot
  output$ground_truth_comparison_plot <- renderPlot({
    req(values$data)
    
    df <- data.frame(
      Math = values$data$true_abilities[, 1],
      Verbal = values$data$true_abilities[, 2]
    )
    
    ggplot(df, aes(x = Math, y = Verbal)) +
      geom_point(alpha = 0.6, color = "#22c55e", size = 2) +
      labs(title = "Ground Truth Factors",
           x = "True Mathematical Ability", 
           y = "True Verbal Ability") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
      coord_equal()
  })
  
  # SVD recovered plot
  output$svd_recovered_plot <- renderPlot({
    req(values$svd_result)
    
    df <- data.frame(
      Factor1 = values$svd_result$U[, 1],
      Factor2 = values$svd_result$U[, 2]
    )
    
    ggplot(df, aes(x = Factor1, y = Factor2)) +
      geom_point(alpha = 0.6, color = "#3b82f6", size = 2) +
      labs(title = "SVD Recovered Factors",
           x = "SVD Factor 1", 
           y = "SVD Factor 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
      coord_equal()
  })
}

# Run the application
shinyApp(ui = ui, server = server)

# ============================================================================
# END OF SECTION 3.1: THE ROTATION PROBLEM
# ============================================================================
