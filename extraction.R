library(data.table)
library(batchtools)
library(mlr3batchmark)


# Load registry
reg = loadRegistry("experiments_fi")

# Get completed jobs
ids = findDone(reg = reg)
# ids = c(53, 104, 95)

# Read results
bmr = reduceResultsBatchmark(ids, store_backends = TRUE, reg = reg)
bmr_with_preds = bmr$score(predictions = TRUE)

# Predictions data.table
predictions_list = list()
for (i in seq_len(nrow(bmr_with_preds))) {
  # Get prediction
  pred = bmr_with_preds$prediction_test[[i]]
  pred_dt = as.data.table(pred)
  
  # Get metadata
  learner_id = bmr_with_preds$learner_id[i]
  iteration = bmr_with_preds$iteration[i]
  task = bmr_with_preds$task[[i]]
  
  # Extract datetime from task backend using row_ids
  row_ids = pred_dt$row_ids
  datetime_data = task$backend$data(rows = row_ids, cols = "datetime")
  # datetime_data[, datetime := with_tz(datetime_data[, datetime], tzone = "America/New_York")]
  
  # Combine into data.table
  result_dt = data.table(
    datetime = datetime_data$datetime,
    row_ids = pred_dt$row_ids,
    learner_id = learner_id,
    iteration = iteration,
    truth = pred_dt$truth,
    response = pred_dt$response
  )
  
  predictions_list[[i]] = result_dt
}
predictions_dt = rbindlist(predictions_list)

# Extract features from each ResampleResult
# Filter pipeop names from estimation_fi.R branching structure
filter_names = c("filter_jmim", "filter_mrmr", "filter_cmim", "filter_importance")

features_list = list()
for (i in seq_len(nrow(bmr$resample_results))) {
  rr = bmr$resample_results$resample_result[[i]]
  learner_id = rr$learners[[1]]$id

  # Loop through each fold
  for (fold in seq_len(rr$resampling$iters)) {
    learner_orig = rr$learners[[fold]]
    learner = learner_orig

    # Get best filter method from AutoTuner archive
    best_fi_method = NULL
    if (inherits(learner_orig, "AutoTuner") && !is.null(learner_orig$tuning_instance)) {
      archive = as.data.table(learner_orig$tuning_instance$archive)
      if ("filter_branch.selection" %in% names(archive)) {
        best_idx = which.min(archive$regr.mse)
        best_fi_method = archive$filter_branch.selection[best_idx]
      }
    }

    # Handle AutoTuner wrapping to get GraphLearner
    if (inherits(learner, "AutoTuner")) {
      learner = learner$learner
    }

    # Extract from GraphLearner
    if (inherits(learner, "GraphLearner")) {
      # Try each filter pipeop name (branched structure)
      filter_pipeop = NULL
      used_filter = NULL

      for (fname in filter_names) {
        if (fname %in% names(learner$graph$pipeops)) {
          fp = learner$graph$pipeops[[fname]]
          if (!is.null(fp$state) && !is.null(fp$state$features)) {
            filter_pipeop = fp
            used_filter = fname
            break
          }
        }
      }

      if (!is.null(filter_pipeop)) {
        selected_features = filter_pipeop$state$features
        feature_scores = filter_pipeop$state$scores

        # Keep top 100 scores to save space
        if (length(feature_scores) > 100) {
          top_idx = order(feature_scores, decreasing = TRUE)[1:100]
          feature_scores = feature_scores[top_idx]
        }

        features_list[[length(features_list) + 1]] = list(
          learner_id = learner_id,
          fold = fold,
          features = selected_features,
          scores = feature_scores,
          n_features = length(selected_features),
          fi_method = best_fi_method,
          filter_used = used_filter
        )
      }
    }
  }

  cat("Processed learner", i, "/", nrow(bmr$resample_results), "\n")
}


# Convert features_list to data.table
features_dt = rbindlist(lapply(features_list, function(x) {
  data.table(
    learner_id = x$learner_id,
    fold = x$fold,
    n_features = x$n_features,
    fi_method = x$fi_method,
    filter_used = x$filter_used,
    features = list(x$features),
    scores = list(x$scores)
  )
}))

# Summary: Best feature importance method per learner
fi_summary = features_dt[, .(
  most_common_fi = names(sort(table(fi_method), decreasing = TRUE))[1],
  n_folds = .N
), by = learner_id]
print("Best feature importance method per learner:")
print(fi_summary)

# Overall best feature importance method across all learners
overall_fi_summary = features_dt[, .N, by = fi_method][order(-N)]
print("Overall feature importance method frequency:")
print(overall_fi_summary)

# Extract selected features for best performing learner
perf_by_learner = bmr_with_preds[, .(
  mean_mse = mean(regr.mse, na.rm = TRUE)
), by = learner_id][order(mean_mse)]
print("Performance by learner (lower MSE = better):")
print(perf_by_learner)

best_learner = perf_by_learner[1, learner_id]
print(paste("Best learner:", best_learner))

# Get features selected by best learner
best_features_dt = features_dt[learner_id == best_learner]
print(paste("Best FI method for best learner:",
            names(sort(table(best_features_dt$fi_method), decreasing = TRUE))[1]))

# Union of all selected features across folds for best learner
all_selected_features = unique(unlist(best_features_dt$features))
print(paste("Total unique features selected:", length(all_selected_features)))
print(all_selected_features)

# Features selected in ALL folds (intersection)
features_per_fold = best_features_dt$features
common_features = Reduce(intersect, features_per_fold)
print(paste("Features selected in ALL folds:", length(common_features)))
print(common_features)
