library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(boot)

set.seed(123)

permute_sign_flipping_contrasts <- function(data, 
                                            formula,
                                            compute_contrast_fn,
                                            subject_col = "subject",
                                            outcome_col = "zvalue",
                                            n_perms = 10000,
                                            seed = 123) {
  set.seed(seed)

  model_obs <- lmer(formula, data = data, REML = FALSE)
  obs_contrast <- compute_contrast_fn(model_obs)

  if (is.null(names(obs_contrast))) {
    names(obs_contrast) <- "contrast"
  }

  contrast_names <- names(obs_contrast)
  perm_contrasts <- matrix(NA, nrow = n_perms, ncol = length(obs_contrast))
  colnames(perm_contrasts) <- contrast_names

  subjects <- unique(data[[subject_col]])

  for (i in 1:n_perms) {
    # flip signs randomly per subject
    signs <- sample(c(-1, 1), size = length(subjects), replace = TRUE)
    sign_df <- data.frame(subject = subjects, sign = signs)
    colnames(sign_df)[1] <- subject_col  # ensure match for merge

    data_perm <- merge(data, sign_df, by = subject_col)
    data_perm[[paste0(outcome_col, "_flipped")]] <- 
      data_perm[[outcome_col]] * data_perm$sign

    # use flipped outcome
    flipped_formula <- as.formula(
      paste(gsub(outcome_col, paste0(outcome_col, "_flipped"), deparse(formula)),
            collapse = " "))

    # fit permuted model
    model_perm <- tryCatch({
      lmer(flipped_formula, data = data_perm, REML = FALSE)
    }, error = function(e) return(NULL))

    if (!is.null(model_perm)) {
      perm_result <- compute_contrast_fn(model_perm)
      if (is.null(names(perm_result))) names(perm_result) <- "contrast"
      perm_contrasts[i, ] <- perm_result
    }
  }

  # remove any rows with NA
  perm_contrasts <- perm_contrasts[complete.cases(perm_contrasts), , drop = FALSE]

  #p-values (two-sided)
  p_values <- sapply(seq_along(obs_contrast), function(j) {
  (sum(abs(perm_contrasts[, j]) >= abs(obs_contrast[j])) + 1) / (nrow(perm_contrasts) + 1)
})

  result <- data.frame(
    contrast = contrast_names,
    observed = obs_contrast,
    p_value = p_values
  )

  return(result)
}

print('perm 10000')
#Brain To TEM g p
data <- read.csv('./outputs/performance_all_averaged_hcecvmpfcTopg_it32000.csv')
data$brain_region <- as.factor(data$brain_region)
data$cell_freq <- as.factor(data$cell_freq)
data$TEM_component <- as.factor(data$TEM_component)
data$fmri_subject <- as.factor(data$fmri_subject)
data <- data %>%
  group_by(brain_region, fmri_subject, TEM_component, cell_freq) %>%
  summarise(Value = mean(Value), zvalue = mean(zvalue), .groups = "drop")
data$TEM_component <- relevel(data$TEM_component, ref = "p")
data$brain_region <- relevel(data$brain_region, ref = "Hippocampus")
data$cell_freq <- factor(data$cell_freq, levels = c("freq0", "freq1", "freq2", "freq3", "freq4"))
model <- lmer(zvalue ~ brain_region * TEM_component * cell_freq + 
                (1 | fmri_subject)+
                (1 |fmri_subject:brain_region)+
                (1 |fmri_subject:TEM_component)+
                (1 |fmri_subject:cell_freq), 
              data = data)

#HC
#p vs. g
compute_contrasts <- function(model) {
  emm <- emmeans(model, ~ brain_region * TEM_component * cell_freq)
  grid <- emm@grid 
  n_rows <- nrow(grid)
  contrast_pg_hc <- numeric(n_rows)
  for (i in 1:n_rows) {
  component <- grid$TEM_component[i]
  brain_region <- grid$brain_region[i]
  if (brain_region == 'Hippocampus') {
    if (component == "p") {
      contrast_pg_hc[i] <- 1 /5
    } else if (component == "g") {
      contrast_pg_hc[i] <- -1 /5
    }
  }
  else {contrast_pg_hc[i] <- 0}
  }
  contrast_spec <- list("p vs. g" = contrast_pg_hc)
  contrast_results <- contrast(emm, contrast_spec)
  return(summary(contrast_results)$estimate)
}
result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ brain_region * TEM_component * cell_freq + 
                (1 | fmri_subject)+
                (1 |fmri_subject:brain_region)+
                (1 |fmri_subject:TEM_component)+
                (1 |fmri_subject:cell_freq),
  subject_col = 'fmri_subject',
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('HC p vs. g')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- quantile(boot_results$t, c(0.025, 0.975))
cat("Observed Contrast Estimate:\n", observed_contrasts, "\n")
cat("\n95% Confidence Intervals:\n", ci, "\n")

#HC vs. ERC/vmPFC
print('HC vs. ERC/vmPFC')
anova(model)

compute_contrasts <- function(model) {
  emm <- emmeans(model, ~ brain_region * TEM_component * cell_freq)
  lin_contrast <- c(-2, -1, 0, 1, 2)
  cell_freq_lin <- contrast(emm, method = list(cell_freq = lin_contrast),
                            by = c("brain_region", "TEM_component"))

  emm_interaction <- emmeans(cell_freq_lin, ~ brain_region * TEM_component)
  pairs_by_tem <- as.data.frame(summary(pairs(emm_interaction, by = "TEM_component", adjust = "none")))
  pairs_by_brain <- as.data.frame(summary(pairs(emm_interaction, by = "brain_region", adjust = "none")))
  class(pairs_by_tem) <- "data.frame"
  class(pairs_by_brain) <- "data.frame"
  colnames(pairs_by_tem)[colnames(pairs_by_tem) == "TEM_component"] <- "Group"
  colnames(pairs_by_brain)[colnames(pairs_by_brain) == "brain_region"] <- "Group"
  relevant_comparisons <- rbind(pairs_by_tem, pairs_by_brain)
  return(relevant_comparisons$estimate)
}

result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ brain_region * TEM_component * cell_freq + 
                (1 | fmri_subject)+
                (1 |fmri_subject:brain_region)+
                (1 |fmri_subject:TEM_component)+
                (1 |fmri_subject:cell_freq),
  subject_col = 'fmri_subject',
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('HC vs. ERC/vmPFC')
emm <- emmeans(model, ~ brain_region * TEM_component * cell_freq)
lin_contrast <- c(-2, -1, 0, 1, 2)
cell_freq_lin <- contrast(emm, method = list(cell_freq = lin_contrast),
                          by = c("brain_region", "TEM_component"))

emm_interaction <- emmeans(cell_freq_lin, ~ brain_region * TEM_component)
pairs_by_tem <- as.data.frame(summary(pairs(emm_interaction, by = "TEM_component", adjust = "none")))
pairs_by_brain <- as.data.frame(summary(pairs(emm_interaction, by = "brain_region", adjust = "none")))
class(pairs_by_tem) <- "data.frame"
class(pairs_by_brain) <- "data.frame"
colnames(pairs_by_tem)[colnames(pairs_by_tem) == "TEM_component"] <- "Group"
colnames(pairs_by_brain)[colnames(pairs_by_brain) == "brain_region"] <- "Group"
relevant_comparisons <- rbind(pairs_by_tem, pairs_by_brain)
print(relevant_comparisons)
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- apply(boot_results$t, 2, function(x) quantile(x, c(0.025, 0.975)))
cat("Observed Contrast Estimates:\n")
print(observed_contrasts)
cat("\n95% Confidence Intervals:\n")
print(ci)

#apHC To p freqs
data <- read.csv('./outputs/performance_all_averaged_apHCtopfreq01freq234_it32000.csv')
data$brain_region <- as.factor(data$brain_region)
data$TEM_freq <- as.factor(data$TEM_freq)
data$cell_freq <- as.factor(data$cell_freq)
data$fmri_subject <- as.factor(data$fmri_subject)
data <- data %>%
  group_by(brain_region, fmri_subject, TEM_component, cell_freq) %>%
  summarise(Value = mean(Value), zvalue = mean(zvalue), .groups = "drop")
model <- lmer(zvalue ~ brain_region * cell_freq + 
                (1 | fmri_subject)+
                (1 |fmri_subject:cell_freq)+
                (1 |fmri_subject:brain_region), data = data)

compute_contrasts <- function(model) {
  emm_interaction <- emmeans(model, ~ brain_region * cell_freq)
  
  low_freq <- c("freq2", "freq3", "freq4")
  high_freq <- setdiff(unique(emm_interaction@grid$cell_freq), low_freq)
  
  grid <- emm_interaction@grid
  n_rows <- nrow(grid)
  contrast_aph_vs_phc_TEM <- numeric(n_rows)

  for (i in 1:n_rows) {
    freq <- grid$cell_freq[i]
    region <- grid$brain_region[i]
    
    if (freq %in% low_freq) {
      if (region == "anteriorHippocampus") {
        contrast_aph_vs_phc_TEM[i] <- 1 / length(low_freq)
      } else if (region == "posteriorHippocampus") {
        contrast_aph_vs_phc_TEM[i] <- -1 / length(low_freq)
      }
    } else if (freq %in% high_freq) {
      if (region == "anteriorHippocampus") {
        contrast_aph_vs_phc_TEM[i] <- -1 / length(high_freq)
      } else if (region == "posteriorHippocampus") {
        contrast_aph_vs_phc_TEM[i] <- 1 / length(high_freq)
      }
    }
  }
  contrast_spec <- list("TEM freq × Region" = contrast_aph_vs_phc_TEM)
  return(summary(contrast(emm_interaction, contrast_spec))$estimate)
}

result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ brain_region * cell_freq + 
                (1 | fmri_subject)+
                (1 |fmri_subject:cell_freq)+
                (1 |fmri_subject:brain_region),
  subject_col = 'fmri_subject',
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('posterior vs anterior x TEM freqs')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- quantile(boot_results$t, c(0.025, 0.975))
cat("Observed Contrast Estimate:\n", observed_contrasts, "\n")
cat("\n95% Confidence Intervals:\n", ci, "\n")

#brain to ratings explained by TEM
data <- read.csv('./outputs/performance_partial_corr_brainToTEMtoRatings_it32000.csv')
data$region <- as.factor(data$region)
data$corr_type <- as.factor(data$corr_type)
data$TEM_component <- as.factor(data$TEM_component)
data$Emotion <- as.factor(data$Emotion)
data$subject <- as.factor(data$subject)
data <- subset(data, region == "Hippocampus")
data <- subset(data, TEM_component == "p")
model <- lmer(zvalue ~ corr_type*Emotion + (1 | subject)+(1 | subject:Emotion)+(1 | subject:corr_type), data = data)

compute_contrasts <- function(model) {
  emm <- emmeans(model, ~ Emotion * corr_type)
  valence_emotions <- c("Good", "Bad", "Calm", "AtEase")
  category_emotions <- setdiff(unique(emm@grid$Emotion), valence_emotions)
  grid <- emm@grid
  n_rows <- nrow(grid)
  contrast_hc <- numeric(n_rows)
  for (i in 1:n_rows) {
    Emotion <- grid$Emotion[i]
    corr_type <- grid$corr_type[i]
    
    if (Emotion %in% category_emotions) {
      if (corr_type == "corr") {
        contrast_hc[i] <- 1 / length(category_emotions)
      } else if (corr_type == "partial_corr") {
        contrast_hc[i] <- -1 / length(category_emotions)
      }
    }
    
    else if (Emotion %in% valence_emotions) {
      if (corr_type == "corr") {
        contrast_hc[i] <- -1 / length(valence_emotions)
      } else if (corr_type == "partial_corr") {
        contrast_hc[i] <- 1 / length(valence_emotions)
      }
    }
  }
  
  contrast_spec <- list("Partial Corr vs. Corr x Category vs. Valence-Arousal" = contrast_hc)
  
  contrast_results <- contrast(emm, contrast_spec)
  return(summary(contrast_results)$estimate)
}

result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ corr_type*Emotion + (1 | subject)+(1 | subject:Emotion)+(1 | subject:corr_type),
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('corr vs partial corr x category vs valence-arousal')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- apply(boot_results$t, 2, function(x) quantile(x, c(0.025, 0.975)))
cat("Observed Contrast Estimates:\n")
print(observed_contrasts)
cat("\n95% Confidence Intervals:\n")
print(ci)


# HC, ERC, vmPFC predicting ratings**
data <- read.csv('./outputs/performance_all_HCECvmPF.csv')
data$region <- as.factor(data$region)
data$Emotion <- as.factor(data$Emotion)
data$emotion_type <- as.factor(data$emotion_type)
data$subject <- as.factor(data$subject)
model <- lmer(zvalue ~ region * Emotion + (1 | subject)+ (1 | subject:Emotion)+(1 | subject:region), data = data)

# HC category vs valence-arousal contrast
compute_contrasts <- function(model) {
  emm_interaction <- emmeans(model, ~ region * Emotion)
  valence_emotions <- c("Good", "Bad", "Calm", "AtEase")
  category_emotions <- setdiff(unique(emm_interaction@grid$Emotion), valence_emotions) 
  grid <- emm_interaction@grid  # the grid of factor levels
  n_rows <- nrow(grid)
  contrast_hc <- numeric(n_rows)
  
  #contrast
  for (i in 1:n_rows) {
    emotion <- grid$Emotion[i]
    region <- grid$region[i]
    
    # for HC 
    if (emotion %in% category_emotions) {
      if (region == "Hippocampus") {
        contrast_hc[i] <- 1/length(category_emotions)
      } else {
        contrast_hc[i] <- 0
      }
    } else if (emotion %in% valence_emotions) {
      if (region == "Hippocampus") {
        contrast_hc[i] <- -1/length(valence_emotions)
      } else {
        contrast_hc[i] <- 0
      }
    }
  }
  contrast_spec <- list("HC's Category vs. Valence-Arousal" = contrast_hc)
  contrast_result <- contrast(emm_interaction, contrast_spec)
  
  return(summary(contrast_result)$estimate) 
}
result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ region * Emotion + (1 | subject) + (1 | subject:Emotion) + (1 | subject:region),
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('HC category vs valence-arousal contrast')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- quantile(boot_results$t, c(0.025, 0.975))
cat("Observed Contrast Estimate:\n")
print(observed_contrasts)
cat("\n95% Confidence Interval:\n")
print(ci)

#HC vs ERC and HC vs vmPFC
compute_contrasts <- function(model) {
  emm_interaction <- emmeans(model, ~ region * Emotion)
  valence_emotions <- c("Good", "Bad", "Calm", "AtEase")
  category_emotions <- setdiff(unique(emm_interaction@grid$Emotion), valence_emotions)
  grid <- emm_interaction@grid  # the grid of factor levels
  n_rows <- nrow(grid)
  contrast_hc_vs_erc <- numeric(n_rows)
  contrast_hc_vs_vmpfc <- numeric(n_rows)
  
  for (i in 1:n_rows) {
    emotion <- grid$Emotion[i]
    region <- grid$region[i]
    
    # HC vs. ERC
    if (emotion %in% category_emotions) {
      if (region == "Hippocampus") {
        contrast_hc_vs_erc[i] <- 1 / length(category_emotions)
      } else if (region == "EntorhinalCortex") {
        contrast_hc_vs_erc[i] <- -1 / length(category_emotions)
      }
    } else if (emotion %in% valence_emotions) {
      if (region == "Hippocampus") {
        contrast_hc_vs_erc[i] <- -1 / length(valence_emotions)
      } else if (region == "EntorhinalCortex") {
        contrast_hc_vs_erc[i] <- 1 / length(valence_emotions)
      }
    }
    
    # HC vs. vmPFC
    if (emotion %in% category_emotions) {
      if (region == "Hippocampus") {
        contrast_hc_vs_vmpfc[i] <- 1 / length(category_emotions)
      } else if (region == "vmPFC_a24_included") {
        contrast_hc_vs_vmpfc[i] <- -1 / length(category_emotions)
      }
    } else if (emotion %in% valence_emotions) {
      if (region == "Hippocampus") {
        contrast_hc_vs_vmpfc[i] <- -1 / length(valence_emotions)
      } else if (region == "vmPFC_a24_included") {
        contrast_hc_vs_vmpfc[i] <- 1 / length(valence_emotions)
      }
    }
  }
  
    contrast_spec <- list(
    "HC vs ERC x Category vs. Valence-Arousal" = contrast_hc_vs_erc,
    "HC vs vmPFC x Category vs. Valence-Arousal" = contrast_hc_vs_vmpfc)
  
  contrast_results <- contrast(emm_interaction, contrast_spec)
  estimates <- c("HC_vs_ERC" = summary(contrast_results)$estimate[1],
                 "HC_vs_vmPFC" = summary(contrast_results)$estimate[2])
  return(estimates)
}
result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ region * Emotion + (1 | subject) + (1 | subject:Emotion) + (1 | subject:region),
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('HC vs ERC and HC vs vmPFC')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- apply(boot_results$t, 2, function(x) quantile(x, c(0.025, 0.975)))
cat("Observed Contrast Estimates:\n")
print(observed_contrasts)
cat("\n95% Confidence Intervals:\n")
print(ci)

#apHC predicting ratings
data <- read.csv('./outputs/performance_all_apHC.csv')
data$region <- as.factor(data$region)
data$emotion_type <- as.factor(data$emotion_type)
data$Emotion <- as.factor(data$Emotion)  
data$subject <- as.factor(data$subject)
model <- lmer(zvalue ~ region * Emotion + (1 | subject)+(1 | subject:Emotion)+(1 | subject:region), data = data)

compute_contrasts <- function(model) {
  emm_interaction <- emmeans(model, ~ region * Emotion)
  valence_emotions <- c("Good", "Bad", "Calm", "AtEase")
  category_emotions <- setdiff(unique(emm_interaction@grid$Emotion), valence_emotions)
  
  grid <- emm_interaction@grid
  n_rows <- nrow(grid)
  contrast_aph_vs_phc <- numeric(n_rows)

  for (i in 1:n_rows) {
    emotion <- grid$Emotion[i]
    region <- grid$region[i]
    
    if (emotion %in% category_emotions) {
      if (region == "anteriorHippocampus") {
        contrast_aph_vs_phc[i] <- -1 / length(category_emotions)
      } else if (region == "posteriorHippocampus") {
        contrast_aph_vs_phc[i] <- 1 / length(category_emotions)
      }
    } else if (emotion %in% valence_emotions) {
      if (region == "anteriorHippocampus") {
        contrast_aph_vs_phc[i] <- 1 / length(valence_emotions)
      } else if (region == "posteriorHippocampus") {
        contrast_aph_vs_phc[i] <- -1 / length(valence_emotions)
      }
    }
  }
  contrast_spec <- list("Category vs Valence-Arousal × Region" = contrast_aph_vs_phc)
  contrast_results <- contrast(emm_interaction, contrast_spec)
  return(summary(contrast_results)$estimate)
}
result <- permute_sign_flipping_contrasts(
  data = data,
  formula = zvalue ~ region * Emotion + (1 | subject) + (1 | subject:Emotion) + (1 | subject:region),
  compute_contrast_fn = compute_contrasts, 
  n_perms = 10000
)
print('posterior vs anterior x category vs valence-arousal')
print(result)
observed_contrasts <- compute_contrasts(model)
boot_results <- bootMer(model, FUN = compute_contrasts, nsim = 10000, type = "parametric")
ci <- quantile(boot_results$t, c(0.025, 0.975))
cat("Observed Contrast Estimate:\n", observed_contrasts, "\n")
cat("\n95% Confidence Intervals:\n", ci, "\n")



