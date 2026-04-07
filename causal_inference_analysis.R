## Paper 3: Optimized TMLE-BART with Unified XGBoost SDG & Sensitivity Analysis
## Methodology: Integrated MICE-XGBoost + Synthetic Augmentation + Robustness Checks
## Updated: 18/02/2026 - External Weight Integration (2024) & Pre-Analysis Diagnostics

## ==============================================================================
## PAPER 3: UNIFIED PATE ESTIMATION (TMLE-BART)
## Methodology: Bootstrap-Integrated mixgb Imputation & XGBoost SDG
## ==============================================================================
## FEATURE                    | PREVIOUS ANALYSIS | UPDATED ANALYSIS (UNIFIED)
## ---------------------------|-------------------|-----------------------------
## MICE Imputation            | Standard RF       | XGBoost (mixgb)
## N-back SDG Generation      | Standalone        | Integrated + Year Alignment
## Uncertainty                | Sampling Error    | Full Propagation (Bootstrap)
## Temporal Logic             | Ignores 15yr Gap  | Repeated Cross-Sectional
## ==============================================================================

# 0. SETUP & LIBRARIES (Robust Installation Logic)
# ------------------------------------------------------------------------------
# STABILITY FIX: Force single-threading for all underlying C++ libraries (OpenMP)
# This prevents the "Session Aborted" crash common in ML loops.

Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(OMP_NUM_THREADS = "1")
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600) 

packages <- c("tidyverse", "dbarts", "xgboost", "mixgb", "magrittr", "mice")
for (pkg in packages) { 
  if(!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
}
library(magrittr)
library(tidyverse)

# 1. LOAD DATA & INITIAL CLEANING
# ------------------------------------------------------------------------------
cat(">>> Stage 1: Loading Data...\n")
if (!file.exists("iso_df_v2.csv")) stop("CRITICAL ERROR: 'iso_df_v2.csv' not found.")

iso_df_v2 <- read.csv("iso_df_v2.csv") %>%
  mutate(across(c(weighting_factor, bmi, dprime2, dprime3, CalorieTotal), ~as.numeric(as.character(.))),
         weighting_factor = ifelse(dataset == "RCT", 1, weighting_factor),
         across(c(Age, sex, ethnicity, country, trying_to_lose_weight, 
                  children_in_household, education_level, group, year, dataset), as.factor))

# 2. SETTINGS & VARIABLE ASSIGNMENTS
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
batch_num <- if(length(args) > 0) as.numeric(args[1]) else 1
set.seed(12345 + batch_num) 

# CLUSTER SETTING: 50 iterations per batch
n_boot      <- 50   
bart_ntree  <- 200
bart_ndpost <- 1000
bart_nskip  <- 250

base_vars        <- c("Age", "sex", "ethnicity") 
full_impute_vars <- c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "year")
gen_vars         <- c("Age", "sex", "ethnicity", "education_level", "bmi", "year")

# ==============================================================================
# STAGE 2: GLOBAL WEIGHT ADJUSTMENT (ACKERMAN POOLING)
# ==============================================================================
cat(">>> Stage 2: Global Weight Adjustment (2008-2023)...\n")
df_proc <- iso_df_v2 %>%
  mutate(year_num = as.numeric(as.character(year)),
         w_group = case_when(dataset == "RCT" ~ "RCT",
                             year_num <= 2011 ~ "Y1_4", year_num <= 2013 ~ "Y5_6",
                             year_num <= 2015 ~ "Y7_8", year_num <= 2018 ~ "Y9_11",
                             TRUE ~ "Y12_15"))

w_sums <- df_proc %>% filter(dataset == "NDNS") %>% group_by(w_group) %>% 
  summarise(sum_w = sum(weighting_factor, na.rm = TRUE), .groups = 'drop')

total_ndns_w <- sum(w_sums$sum_w)
w_sums %<>% mutate(temporal_fraction = case_when(w_group == "Y1_4" ~ 4/16, w_group == "Y5_6" ~ 2/16, 
                                                 w_group == "Y7_8" ~ 2/16, w_group == "Y9_11" ~ 3/16, 
                                                 TRUE ~ 5/16))

df_proc <- df_proc %>% left_join(w_sums, by = "w_group") %>%
  mutate(final_weight = case_when(dataset == "RCT" ~ 1,
                                  !is.na(sum_w) & sum_w > 0 ~ weighting_factor * (total_ndns_w / sum_w) * temporal_fraction,
                                  TRUE ~ 1)) %>%
  mutate(final_weight = ifelse(dataset == "NDNS", final_weight / mean(final_weight[dataset == "NDNS"], na.rm = TRUE), 1))

# ==============================================================================
# STAGE 3: UNIFIED ANALYSIS LOOP (MICE, SDG, DR-BART)
# ==============================================================================
run_unified_analysis <- function() {
  
  ndns_data <- df_proc %>% filter(dataset == "NDNS")
  rct_data  <- df_proc %>% filter(dataset == "RCT")
  y_min <- min(df_proc$CalorieTotal, na.rm = TRUE)
  y_max <- max(df_proc$CalorieTotal, na.rm = TRUE)
  
  # NEW: Expand the tracking list to include the 12 specific Model Scenarios
  model_names <- c(
    "Model A", "Model B", "Model C", "Model D",
    "Model E_Base", "Model E_Lower", "Model E_Upper", "Model E_Linear",
    "Model F_Base", "Model F_Lower", "Model F_Upper", "Model F_Linear"
  )
  
  results_all_models <- lapply(model_names, function(x) {
    res <- matrix(NA, nrow = n_boot, ncol = 3)
    colnames(res) <- c("Coarse_v_Control", "Detailed_v_Control", "Coarse_v_Detailed")
    return(res)
  })
  names(results_all_models) <- model_names
  
  for(i in 1:n_boot) {
    cat(paste("--- Unified Iteration", i, "---\n"))
    
    # --- STAGE 1: BOOTSTRAP ---
    boot_ndns <- ndns_data %>% slice_sample(n = nrow(ndns_data), replace = TRUE, weight_by = final_weight)
    boot_rct  <- rct_data %>% slice_sample(n = nrow(rct_data), replace = TRUE)
    
    
     # --- STAGE 2: GLOBAL IMPUTATION (mixgb) ---
    cat("    ... Running Global Imputation (mixgb - Thread Locked)\n")
    gc() 
    
    # 1. Combine bootstrapped samples and explicitly create 'in_trial' 
    boot_all_unimp <- bind_rows(boot_rct, boot_ndns) %>%
      mutate(in_trial = ifelse(dataset == "RCT", 1, 0))
    
    # 2. Select variables mimicking your old RF approach
    imp_input <- boot_all_unimp %>% 
      select(all_of(full_impute_vars), CalorieTotal, dataset, in_trial, group, final_weight) %>%
      select(where(~ !all(is.na(.)))) # Drop empty columns just in case
    
    # 3. Impute globally using XGBoost (CRITICAL FIX: xgb.params locks the threads)
    imp_res <- mixgb::mixgb(
      data = imp_input, 
      m = 1, 
      xgb.params = list(nthread = 1, max_depth = 3), 
      verbose = FALSE
    )
    
    # 4. CRITICAL FIX: Reattach the dropped cognitive scores before filtering!
    df_clean_global <- imp_res[[1]]
    df_clean_global$dprime2 <- boot_all_unimp$dprime2
    df_clean_global$dprime3 <- boot_all_unimp$dprime3
    
    # Clean and filter exactly like the old code
    df_clean_global <- df_clean_global %>%
      filter(in_trial == 0 | (in_trial == 1 & !is.na(CalorieTotal)))
    
    # 5. Split back out for Stage 3 (Synthetic Data Generation)
    boot_rct_clean  <- df_clean_global %>% filter(in_trial == 1)
    boot_ndns_clean <- df_clean_global %>% filter(in_trial == 0)
    
    # Free up RAM
    rm(imp_res, imp_input, boot_all_unimp, df_clean_global)
    gc()
    
   
    
    # --- STAGE 3: SDG (XGBoost) ---
    # ==============================================================================
    # --- STAGE 3: SYNTHETIC DATA GENERATION (Dual-Engine with ELSA Constraints) ---
    # Primary: XGBoost (Base Traits) + Deterministic ELSA Penalties + Noise
    # Appendix: Piecewise Linear Model + Deterministic ELSA Penalties + Noise
    # ==============================================================================
    cat("    ... Generating Synthetic N-back Scores (Dual-Engine + ELSA Penalties)\n")
    
    # --------------------------------------------------------------------------
    # 1. PREPARE DATA & ENSURE NUMERIC AGE
    # --------------------------------------------------------------------------
    # We must convert categorical age brackets into mathematical midpoints so we 
    # can calculate precise, year-by-year cognitive decline penalties.
    convert_age <- function(df) {
      df %>% mutate(
        Age_Clean = str_trim(as.character(Age)),
        Age_num = case_when(
          Age_Clean == "18-24"       ~ 21,
          Age_Clean == "25-34"       ~ 29.5,
          Age_Clean == "35-44"       ~ 39.5,
          Age_Clean == "45-54"       ~ 49.5, 
          Age_Clean == "55-64"       ~ 59.5, 
          Age_Clean == "65 and over" ~ 75,   
          TRUE ~ NA_real_ 
        )
      )
    }
    
    # Apply the conversion to both the RCT and the NDNS bootstrap samples
    boot_rct_clean  <- convert_age(boot_rct_clean)
    boot_ndns_clean <- convert_age(boot_ndns_clean)
    
    # Calculate exact d'2 shift for THIS specific bootstrap sample.
    # By subtracting the means rather than using a hardcoded multiplier (like * 0.9), 
    # we perfectly preserve the variance and mathematical relationship of the trial data.
    d2_shift <- mean(boot_rct_clean$dprime2, na.rm=TRUE) - mean(boot_rct_clean$dprime3, na.rm=TRUE)
    
    # --------------------------------------------------------------------------
    # 2. CALCULATE STRICT BIOLOGICAL PENALTIES (Based on ELSA Literature)
    # --------------------------------------------------------------------------
    # Based on the English Longitudinal Study of Ageing (ELSA):
    # - Ages < 55: Cognition remains stable (0 penalty).
    # - Ages 55-74: 10% decline overall. We spread this -0.225 drop smoothly 
    #               across 20 years (-0.01125 per year past age 54).
    # - Ages 75+: Steeper 24% decline. We apply the previous cap (-0.225) and add 
    #             -0.054 per year past age 74.
    # - Education: Lower education significantly widens the gap at older ages.
    
    boot_ndns_clean <- boot_ndns_clean %>%
      mutate(
        elsa_age_penalty = case_when(
          Age_num < 55 ~ 0,  
          
          # This ensures the 59.5 midpoint receives a proportionate penalty 
          # rather than being artificially treated as "stable".
          Age_num >= 55 & Age_num < 75 ~ (Age_num - 54) * -0.01125, 
          
          # Starts at the -0.225 cap from the previous bracket, then accelerates
          Age_num >= 75 ~ (-0.225) + ((Age_num - 74) * -0.054),    
          
          TRUE ~ 0
        ),
        
        # Applies an extra penalty to older demographics lacking higher education
        elsa_edu_penalty = ifelse(Age_num >= 75 & education_level == "Low", -0.15, 0)
      )
    
    # --------------------------------------------------------------------------
    # 3. PRIMARY ENGINE: XGBoost (Base Traits) + ELSA Penalties
    # --------------------------------------------------------------------------
    # We explicitly REMOVE Age_num from XGBoost training. XGBoost learns the 
    # baseline "fingerprint" of Sex, Ethnicity, and Education from the RCT.
    train_df_xgb <- boot_rct_clean %>% select(dprime3, all_of(setdiff(gen_vars, "Age"))) %>% drop_na()
    train_mx_xgb <- model.matrix(dprime3 ~ . - 1, data = train_df_xgb)
    
    # Train the machine learning model (Thread Locked for cluster stability)
    bst_gen <- xgboost(
      x = as.matrix(train_mx_xgb), 
      y = as.numeric(train_df_xgb$dprime3), 
      nrounds = 100, 
      objective = "reg:squarederror", 
      max_depth = 3, 
      nthread = 1
    )
    
    # Prepare NDNS matrix and align columns to prevent prediction crashes
    ndns_mx_xgb <- model.matrix(~ . - 1, data = boot_ndns_clean %>% select(all_of(setdiff(gen_vars, "Age"))))
    missing_gen <- setdiff(colnames(train_mx_xgb), colnames(ndns_mx_xgb))
    if(length(missing_gen) > 0) {
      ndns_mx_xgb <- cbind(ndns_mx_xgb, matrix(0, nrow=nrow(ndns_mx_xgb), ncol=length(missing_gen), dimnames=list(NULL, missing_gen)))
    }
    
    # Predict XGB Base, then add the deterministic ELSA constraints and natural noise
    xgb_base_pred <- predict(bst_gen, newdata = as.matrix(ndns_mx_xgb[, colnames(train_mx_xgb)]), nthread = 1)
    xgb_final     <- xgb_base_pred + boot_ndns_clean$elsa_age_penalty + boot_ndns_clean$elsa_edu_penalty + rnorm(nrow(boot_ndns_clean), 0, 0.5)
    
    # Save XGBoost Scenarios (Base, Lower -10%, Upper +10%)
    boot_ndns_clean$dprime3       <- xgb_final
    boot_ndns_clean$dprime2       <- boot_ndns_clean$dprime3 + d2_shift
    
    boot_ndns_clean$dprime3_lower <- xgb_final * 0.90
    boot_ndns_clean$dprime3_upper <- xgb_final * 1.10
    boot_ndns_clean$dprime2_lower <- boot_ndns_clean$dprime3_lower + d2_shift
    boot_ndns_clean$dprime2_upper <- boot_ndns_clean$dprime3_upper + d2_shift
    
    # --------------------------------------------------------------------------
    # 4. APPENDIX ENGINE: Piecewise Linear Model + ELSA Penalties
    # --------------------------------------------------------------------------
    # CRITICAL FIX: To prevent "new factor level" crashes when the RCT bootstrap omits 
    # rare categories (like "Primary school"), we train the linear model on the exact 
    # same aligned dummy-variable matrices we safely built for XGBoost.
    
    train_df_lin <- as.data.frame(train_mx_xgb)
    train_df_lin$dprime3 <- train_df_xgb$dprime3
    
    # Train standard linear regression on the aligned dummy variables
    fit_d3_lin   <- lm(dprime3 ~ ., data = train_df_lin)
    sigma_d3_lin <- sigma(fit_d3_lin)
    
    # Predict using the safely aligned NDNS matrix
    ndns_df_lin   <- as.data.frame(ndns_mx_xgb[, colnames(train_mx_xgb)])
    lin_base_pred <- predict(fit_d3_lin, newdata = ndns_df_lin)
    
    # Assemble the final linear predictions with identical penalties and noise
    boot_ndns_clean$dprime3_linear <- lin_base_pred + boot_ndns_clean$elsa_age_penalty + boot_ndns_clean$elsa_edu_penalty + rnorm(nrow(boot_ndns_clean), 0, sigma_d3_lin)
    boot_ndns_clean$dprime2_linear <- boot_ndns_clean$dprime3_linear + d2_shift
    
    # ==============================================================================
    # --- DIAGNOSTIC OUTPUT: Print Validation Tables on Iteration 1 Only ---
    # ==============================================================================
    if (i == 1) {
      cat("\n====================================================================\n")
      cat(">>> DIAGNOSTIC OUTPUT (Iteration 1: Synthetic Data Validation) <<<\n")
      cat("====================================================================\n")
      
      # 1. Combine datasets to build the tables
      diag_df <- bind_rows(
        boot_rct_clean %>% mutate(dataset = "RCT"),
        boot_ndns_clean %>% mutate(dataset = "NDNS") # NDNS now has the synthetic d' scores
      ) %>%
        mutate(Age_Clean = str_trim(as.character(Age)))
      
      # ----------------------------------------------------------------------
      # TABLE 1: Cognitive Scores by Age Bracket (Total vs RCT vs NDNS)
      # ----------------------------------------------------------------------
      age_levels <- c("18-24", "25-34", "35-44", "45-54", "55-64", "65 and over")
      
      table_age <- diag_df %>%
        mutate(Age_Clean = factor(Age_Clean, levels = age_levels)) %>%
        group_by(Age_Clean, dataset) %>%
        summarise(
          mean_d2 = mean(dprime2, na.rm = TRUE),
          mean_d3 = mean(dprime3, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        pivot_wider(names_from = dataset, values_from = c(mean_d2, mean_d3)) %>%
        # Calculate the "Total" column combining both datasets
        left_join(
          diag_df %>%
            mutate(Age_Clean = factor(Age_Clean, levels = age_levels)) %>%
            group_by(Age_Clean) %>%
            summarise(
              mean_d2_Total = mean(dprime2, na.rm = TRUE),
              mean_d3_Total = mean(dprime3, na.rm = TRUE),
              .groups = "drop"
            ),
          by = "Age_Clean"
        ) %>%
        # Clean up column names for a beautiful console output
        select(
          `Age Bracket` = Age_Clean,
          `Total d'2`   = mean_d2_Total,
          `Total d'3`   = mean_d3_Total,
          `RCT d'2`     = mean_d2_RCT,
          `RCT d'3`     = mean_d3_RCT,
          `NDNS d'2`    = mean_d2_NDNS,
          `NDNS d'3`    = mean_d3_NDNS
        ) %>%
        arrange(`Age Bracket`) %>%
        mutate(across(where(is.numeric), ~round(., 2))) # Round to 2 decimals
      
      cat("\n--- TABLE 1: Mean d'2 and d'3 by Age Bracket ---\n")
      print(as.data.frame(table_age), row.names = FALSE)
      
      # ----------------------------------------------------------------------
      # TABLE 2: Cognitive Scores by Survey Year
      # ----------------------------------------------------------------------
      table_year <- diag_df %>%
        group_by(year, dataset) %>%
        summarise(
          `Mean d'2` = round(mean(dprime2, na.rm = TRUE), 2),
          `Mean d'3` = round(mean(dprime3, na.rm = TRUE), 2),
          `Sample Size` = n(),
          .groups = "drop"
        ) %>%
        arrange(year, dataset) %>%
        rename(`Survey Year` = year, `Dataset` = dataset)
      
      cat("\n--- TABLE 2: Mean d'2 and d'3 by Survey Year ---\n")
      print(as.data.frame(table_year), row.names = FALSE)
      cat("====================================================================\n\n")
      

    }
    
    # ==============================================================================
    # --- STAGE 4: DR-BART (TMLE) WITH SENSITIVITY NESTING ---
    # ==============================================================================
    cat("    ... Estimating Nested PATEs (Main + Sensitivity)\n")
    
    # 1. Prepare Trial Data (Ground truth is fixed across all sensitivity scenarios)
    boot_rct_prepared <- boot_rct_clean %>%
      mutate(
        in_trial = 1,
        dprime2_lower  = dprime2,
        dprime2_upper  = dprime2,
        dprime2_linear = dprime2,
        dprime3_lower  = dprime3,
        dprime3_upper  = dprime3,
        dprime3_linear = dprime3
      )
    
    # 2. Combine with NDNS (which contains the synthetically varied columns)
    boot_all <- bind_rows(boot_rct_prepared, boot_ndns_clean %>% mutate(in_trial = 0))
    
    # 3. Dynamically define the covariate lists for this iteration
    iter_models_list <- list(
      "Model A" = base_vars,
      "Model B" = c(base_vars, "children_in_household"),
      "Model C" = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household"),
      "Model D" = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level"),
      
      "Model E_Base"   = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime2"),
      "Model E_Lower"  = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime2_lower"),
      "Model E_Upper"  = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime2_upper"),
      "Model E_Linear" = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime2_linear"),
      
      "Model F_Base"   = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime3"),
      "Model F_Lower"  = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime3_lower"),
      "Model F_Upper"  = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime3_upper"),
      "Model F_Linear" = c(base_vars, "trying_to_lose_weight", "bmi", "children_in_household", "education_level", "dprime3_linear")
    )
    
    # 4. Execute DR-BART (TMLE) Loop across all 12 model variations
    for(m_name in names(iter_models_list)) {
      covs <- iter_models_list[[m_name]]
      
      # Clean missingness and create matrix safely
      df_model <- boot_all %>% select(all_of(covs), in_trial, group, CalorieTotal) %>% drop_na()
      x_mat <- dbarts::makeModelMatrixFromDataFrame(df_model %>% select(all_of(covs)))
      
      # -- PROPENSITY SCORE (Membership) MODEL --
      ps_bart <- bart(
        x.train = x_mat, y.train = df_model$in_trial, 
        ntree = bart_ntree, ndpost = floor(bart_ndpost/2), nskip = bart_nskip, 
        verbose = FALSE, nthread = 1 # CRITICAL FIX: Removed the dot
      )
      
      # Bound Propensity scores to prevent infinite TMLE weights
      ps_hat <- pmax(pmin(pnorm(colMeans(ps_bart$yhat.train)), 0.99), 0.01)
      rm(ps_bart)
      
      # -- OUTCOME TARGETING FUNCTION (TMLE) --
      target_pate <- function(lA, lB) {
        get_mean <- function(grp) {
          idx_arm <- which(df_model$in_trial == 1 & df_model$group == grp)
          idx_pop <- which(df_model$in_trial == 0)
          
          # Outcome Model (Q0)
          fit_y <- bart(
            x.train = x_mat[idx_arm, ], y.train = df_model$CalorieTotal[idx_arm], 
            x.test = x_mat, ntree = bart_ntree, ndpost = bart_ndpost, 
            verbose = FALSE, nthread = 1 # CRITICAL FIX: Removed the dot
          )
          
          q0 <- fit_y$yhat.test.mean
          h_rct <- 1/ps_hat[idx_arm]
          h_pop <- 1/ps_hat[idx_pop]
          
          # Scale outcomes for bounded logistic regression
          y_s <- (df_model$CalorieTotal[idx_arm] - y_min) / (y_max - y_min)
          q0_rct_s <- (q0[idx_arm] - y_min) / (y_max - y_min)
          q0_pop_s <- (q0[idx_pop] - y_min) / (y_max - y_min)
          
          # Calculate TMLE Epsilon Shift
          eps <- coef(glm(y_s ~ -1 + h_rct + offset(qlogis(pmax(pmin(q0_rct_s, 0.999), 0.001))), family = quasibinomial()))
          
          # Apply shift to target population predictions and unscale
          res_val <- mean(plogis(qlogis(pmax(pmin(q0_pop_s, 0.999), 0.001)) + eps * h_pop) * (y_max - y_min) + y_min)
          rm(fit_y)
          return(res_val)
        }
        return(get_mean(lA) - get_mean(lB))
      }
      
      # Estimate and store the 3 policy contrasts
      results_all_models[[m_name]][i, 1] <- target_pate("Coarse", "No label")
      results_all_models[[m_name]][i, 2] <- target_pate("Detailed", "No label")
      results_all_models[[m_name]][i, 3] <- target_pate("Coarse", "Detailed")
      
      rm(x_mat, df_model)
      gc()
    } # <-- This bracket closes the INNER loop (the 12 models)
    
    # 5. Final memory wipe for this specific Bootstrap Iteration
    rm(boot_rct_prepared, boot_all, boot_ndns_clean, boot_rct_clean)
    gc()
    
  } # <--- THIS BRACKET CLOSES THE 50-ITERATION BOOTSTRAP LOOP. 
  
  # Function output goes here, safely AFTER all loops have finished:
  return(results_all_models)
  
} # <--- This bracket closes the run_unified_analysis() function

# 5. EXECUTION
# ------------------------------------------------------------------------------
final_results <- run_unified_analysis()

# SAVE TO CLUSTER BEFORE PRINTING SUMMARY
saveRDS(final_results, paste0("results_batch_", batch_num, ".rds"))
cat(paste0("\n>>> BATCH ", batch_num, " SECURELY SAVED TO RDS FILE.\n"))

# ==============================================================================
# Generate PATE Summary Table with 95% Confidence Intervals
# ==============================================================================
library(dplyr)

pate_summary_table_ci <- bind_rows(lapply(names(final_results), function(model_name) {
  
  # Extract the matrix for this specific model
  mat <- final_results[[model_name]]
  
  # Helper function to calculate Mean and 95% CI from the bootstrap distribution
  format_estimate <- function(estimates) {
    est_mean <- mean(estimates, na.rm = TRUE)
    est_ci   <- quantile(estimates, probs = c(0.025, 0.975), na.rm = TRUE)
    # Format as "Mean [2.5%, 97.5%]"
    sprintf("%.2f [%.2f, %.2f]", est_mean, est_ci[1], est_ci[2])
  }
  
  # Format into a clean row
  data.frame(
    `Model Hierarchy` = model_name,
    `Coarse vs No Label` = format_estimate(mat[, 1]),
    `Detailed vs No Label` = format_estimate(mat[, 2]),
    `Coarse vs Detailed` = format_estimate(mat[, 3]),
    check.names = FALSE
  )
}))

# Print the table to the console
cat("\n====================================================================\n")
cat(">>> PATE ESTIMATES & 95% CI (Based on", nrow(final_results[[1]]), "Iterations) <<<\n")
cat("====================================================================\n")
print(pate_summary_table_ci, row.names = FALSE)
cat("====================================================================\n")