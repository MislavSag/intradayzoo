library(data.table)
library(batchtools)
library(mlr3batchmark)


# Load registry
experiment_path = "experiments_fi"
reg = loadRegistry("experiments_fi")

# Get completed jobs
ids = findDone(reg = reg)
# ids = c(39, 40, 56)

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
fwrite(predictions_dt, file.path(experiment_path, "predictions.csv"))

# Extract archives and results
# archives = mlr3tuning::extract_inner_tuning_archives(bmr)
# saveRDS(archives, file.path(experiment_path, "archives.rds")) # TOO BIG
results  = mlr3tuning::extract_inner_tuning_results(bmr)
saveRDS(results, file.path(experiment_path, "results.rds"))

# Plot results based on feature importance
if (interactive()) {
  library(ggplot2)
  archives[, filter_branch.selection == x_domain_filter_branch.selection]
  archives[, .(mse = mean(regr.mse)), by = .(fi = filter_branch.selection)] |>
    ggplot(aes(x = fi, y = mse)) +
    geom_bar(stat = "identity")
  archives[, .(mse = median(regr.mse)), by = .(fi = filter_branch.selection)] |>
    ggplot(aes(x = fi, y = mse)) +
    geom_bar(stat = "identity")
  archives[, .(mse = min(regr.mse)), by = .(fi = filter_branch.selection)] |>
    ggplot(aes(x = fi, y = mse)) +
    geom_bar(stat = "identity")
  }

# Extract predictors
extract_inner_tuning_fi.ResampleResult = function(x, tuning_instance = FALSE, ...) {
  # x = bmr$resample_results$resample_result[[1]]
  rr = assert_resample_result(x)
  if (is.null(rr$learners[[1]]$model$tuning_instance)) {
    return(data.table())
  }
  tab = imap_dtr(rr$learners, function(learner, i) {
    # learner = rr$learners[[1]]
    x = learner$model$learner$state$model
    data = as.data.table(x[[length(x)]]$feature_names)
    set(data, j = "iteration", value = i)
    data
  })
  tab[, "task_id" := rr$task$id]
  tab[, "learner_id" := rr$learner$id]
  tab[, "resampling_id" := rr$resampling$id]
  tab
}

extract_inner_tuning_fi.BenchmarkResult = function(x, tuning_instance = FALSE, ...) {
  bmr = assert_benchmark_result(x)
  tab = imap_dtr(bmr$resample_results$resample_result, function(rr, i) {
    # rr = bmr$resample_results$resample_result[[1]]
    data = extract_inner_tuning_fi.ResampleResult(rr, tuning_instance = tuning_instance)
     if (nrow(data) > 0) set(data, j = "experiment", value = i)
  }, .fill = TRUE)
  # reorder dt
  tab
}

fi = extract_inner_tuning_fi.BenchmarkResult(bmr)
fwrite(fi, file.path(experiment_path, "fi.csv"))
