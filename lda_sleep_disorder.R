# -------------------------------------------------------
# Sleep Disorder Classification using LDA
# Course Project: Data Science & Machine Learning
# -------------------------------------------------------

# Load required libraries
library(tidyr)
library(MASS)

# -------------------------------------------------------
# Load Dataset (relative path for GitHub compatibility)
# -------------------------------------------------------
SleepDisorder <- read.csv("data/Sleep_health_and_lifestyle_dataset.csv")

# -------------------------------------------------------
# Data Cleaning & Preprocessing
# -------------------------------------------------------

# Remove ID column
SleepDisorder$Person.ID <- NULL

# Convert categorical variables to factors
SleepDisorder$Sleep.Disorder <- as.factor(SleepDisorder$Sleep.Disorder)
SleepDisorder$Gender <- as.factor(SleepDisorder$Gender)
SleepDisorder$Occupation <- as.factor(SleepDisorder$Occupation)
SleepDisorder$BMI.Category <- as.factor(SleepDisorder$BMI.Category)

# Split Blood Pressure into Systolic and Diastolic
SleepDisorder <- separate(
  SleepDisorder,
  col = Blood.Pressure,
  into = c("Systolic", "Diastolic"),
  sep = "/"
)

SleepDisorder$Systolic  <- as.numeric(SleepDisorder$Systolic)
SleepDisorder$Diastolic <- as.numeric(SleepDisorder$Diastolic)

# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------

num_vars <- c(
  "Age", "Sleep.Duration", "Quality.of.Sleep",
  "Physical.Activity.Level", "Stress.Level",
  "Systolic", "Diastolic", "Heart.Rate",
  "Daily.Steps"
)

SleepDisorder[, num_vars] <- scale(SleepDisorder[, num_vars])

# -------------------------------------------------------
# Full LDA Model
# -------------------------------------------------------

lda_full <- lda(
  Sleep.Disorder ~ Gender + Age + Occupation + Sleep.Duration +
    Quality.of.Sleep + Physical.Activity.Level + Stress.Level +
    BMI.Category + Systolic + Diastolic + Heart.Rate + Daily.Steps,
  data = SleepDisorder
)

print(lda_full)

# Variable importance
importance <- abs(lda_full$scaling)
importance <- importance[order(-importance[,1]), , drop = FALSE]
print(importance)

# -------------------------------------------------------
# Model Evaluation (10 Random Train/Test Splits)
# -------------------------------------------------------

set.seed(123)
runs <- 10
labels <- levels(SleepDisorder$Sleep.Disorder)

evaluate_model <- function(formula, data) {
  test_errors <- numeric(runs)
  conf_matrices <- vector("list", runs)
  
  for (i in 1:runs) {
    idx <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
    train <- data[idx, ]
    test  <- data[-idx, ]
    
    model <- tryCatch(
      lda(formula, data = train),
      error = function(e) NULL
    )
    
    if (is.null(model)) next
    
    preds <- predict(model, newdata = test)$class
    test_errors[i] <- mean(preds != test$Sleep.Disorder)
    conf_matrices[[i]] <- table(Predicted = preds, Actual = test$Sleep.Disorder)
  }
  
  avg_cm <- matrix(
    0, length(labels), length(labels),
    dimnames = list(Predicted = labels, Actual = labels)
  )
  
  for (cm in conf_matrices) {
    if (!is.null(cm)) avg_cm <- avg_cm + cm
  }
  
  list(
    mean_error = mean(test_errors),
    avg_confusion_matrix = avg_cm / runs
  )
}

# Full model evaluation
full_results <- evaluate_model(
  Sleep.Disorder ~ Gender + Age + Occupation + Sleep.Duration +
    Quality.of.Sleep + Physical.Activity.Level + Stress.Level +
    BMI.Category + Systolic + Diastolic + Heart.Rate + Daily.Steps,
  SleepDisorder
)

print(full_results)

# -------------------------------------------------------
# Reduced LDA Model
# -------------------------------------------------------

reduced_results <- evaluate_model(
  Sleep.Disorder ~ Occupation + Diastolic + Quality.of.Sleep +
    Systolic + Heart.Rate + BMI.Category + Age + Sleep.Duration,
  SleepDisorder
)

print(reduced_results)
