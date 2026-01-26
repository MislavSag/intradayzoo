library(data.table)
library(batchtools)
library(mlr3batchmark)


# Load registry
reg = loadRegistry("experiments_fi")

# Get completed jobs
ids = findDone(reg = reg)

bmr = reduceResultsBatchmark(1, store_backends = TRUE, reg = reg)
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
features_list = list()

for (i in seq_len(nrow(bmr$resample_results))) {
  rr = bmr$resample_results$resample_result[[i]]
  learner_id = rr$learners[[1]]$id
  
  # Loop through each fold
  for (fold in seq_len(rr$resampling$iters)) {
    learner = rr$learners[[fold]]
    
    # Handle AutoTuner wrapping
    if (inherits(learner, "AutoTuner")) {
      learner = learner$learner
    }
    
    # Extract from GraphLearner
    if (inherits(learner, "GraphLearner")) {
      filter_pipeop = learner$graph$pipeops[["filter"]]
      
      if (!is.null(filter_pipeop) && !is.null(filter_pipeop$state)) {
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
          n_features = length(selected_features)
        )
      }
    }
  }
  
  cat("Processed learner", i, "/", nrow(bmr$resample_results), "\n")
}


bmr$learners$learner[[1]]$
x = bmr$resample_result(1)
x$learner$archive
x = as.data.table(bmr$resample_results)
x$resample_result[[1]]
bmr$learners$learner[[1]]$
