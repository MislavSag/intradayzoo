library(data.table)
library(mlr3)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3misc)
library(mlr3filters)
library(mlr3learners)
library(mlr3extralearners)
library(batchtools)
library(mlr3batchmark)
library(torch)
library(mlr3torch)
library(mlr3finance)
library(paradox)
library(mlr3hyperband)


# TODO:
# 1. datetime is UTC when imported

# Custom PipeOp to filter rows during training but not prediction
PipeOpFilterJumps = R6::R6Class(
  "PipeOpFilterJumps",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  
  public = list(
    initialize = function(id = "filter_jumps") {
      super$initialize(
        id = id,
        param_set = ps(),
        packages = character(0),
        task_type = "TaskRegr"
      )
    }
  ),
  
  private = list(
    .train_task = function(task) {
      # During training: filter out rows where jump == 1
      # Assumes 'jump' column is available in the task backend
      task_data = task$data()
      
      # Get the jump column from backend (it should be in id columns or elsewhere)
      if ("jump" %in% names(task$backend$data(rows = task$row_ids, cols = task$backend$colnames))) {
        jump_data = task$backend$data(rows = task$row_ids, cols = "jump")
        keep_rows = task$row_ids[jump_data$jump == 0]
        task$filter(keep_rows)
      } else {
        warning("'jump' column not found in task backend. No filtering applied.")
      }
      
      task
    },
    
    .predict_task = function(task) {
      # During prediction: keep all rows (do nothing)
      task
    }
  )
)

# Register the custom PipeOp (optional, for convenience)
mlr_pipeops$add("filter_jumps", PipeOpFilterJumps)

# Parameters
JUMP = FALSE

# Import data
factors = fread("data/factor_returns.csv")

# Prepare data
head(colnames(factors)); tail(colnames(factors))
id_columns = c("datetime", "jump")
factors[, year := data.table::year(datetime)]
factors[, .(datetime, jump, target, targetc)]

# Remove missing targets
factors[, .SD, .SDcols = factors[, which(unlist(lapply(.SD, function(x) any(is.na(x)))))]]
factors = na.omit(factors, cols = "target")

# Define tasks
task  = as_task_regr(factors[, .SD, .SDcols = -c("targetc")], "target", "factorszoo")
# taskc = as_task_regr(factors[, .SD, .SDcols = -c("target")], "targetc", "factorszooc")

# set roles for id columns
task$set_col_roles("year", "group")
task$col_roles$feature = setdiff(task$col_roles$feature, id_columns)
# taskc$set_col_roles("year", "group")
# taskc$col_roles$feature = setdiff(taskc$col_roles$feature, id_columns)

# Cross validation resampling parameters
FIRST_YEAR = 2004
train_size_years_init = FIRST_YEAR - factors[, min(year)]

# Test rolling cross validation
if (interactive()) {
  library(ggplot2)
  library(patchwork)

  r = ResamplingGapCV$new()
  r$param_set$values
  r$param_set$values$initial_window = train_size_years_init + 7
  r$param_set$values$horizon = 1
  r$param_set$values$gap = 0
  r$param_set$values$step = 1
  r$param_set$values$rolling = FALSE
  r$instantiate(task)

  prepare_cv_plot = function(x, set = "train") {
    x = lapply(x, function(x) data.table(ID = x))
    x = rbindlist(x, idcol = "fold")
    x[, fold := as.factor(fold)]
    x[, set := as.factor(set)]
    x[, ID := as.numeric(ID)]
  }

  n_ = factors[, length(unique(year))] - (train_size_years_init + 7) # 10
  trains = lapply(1:n_, function(x) r$train_set(x))
  tests = lapply(1:n_, function(x) r$test_set(x))
  trains = prepare_cv_plot(trains, set = "train")
  tests = prepare_cv_plot(tests, set = "test")
  dt_vis = rbind(trains, tests)
  ggplot(dt_vis, aes(x = fold, y = ID, color = set)) +
    geom_point() +
    theme_minimal() +
    coord_flip() +
    labs(x = "", y = '')
}

# Create autotuners
create_autotuner = function(
  learner,
  tuner = tnr("hyperband", eta = 6),
  search_space = NULL,
  n_evals = 3,
  hyper = TRUE,
  filter_jumps = JUMP) {  # Add parameter to reference global JUMP

# Create branching filter graph - tuner chooses best filter method
  # Fixed filter.frac = 0.02 (2% of features) to compare methods fairly
  filter_graph = po("branch", id = "filter_branch",
                    options = c("jmim", "mrmr", "cmim", "importance")) %>>%
    gunion(list(
      jmim = po("filter", id = "filter_jmim", filter = flt("jmim"), filter.frac = 0.02),
      mrmr = po("filter", id = "filter_mrmr", filter = flt("mrmr"), filter.frac = 0.02),
      cmim = po("filter", id = "filter_cmim", filter = flt("cmim"), filter.frac = 0.02),
      importance = po("filter", id = "filter_importance",
                      filter = flt("importance", learner = lrn("regr.ranger", num.trees = 100)),
                      filter.frac = 0.02)
    )) %>>%
    po("unbranch", id = "filter_unbranch")

  # Apply jump filter if needed (before feature selection)
  if (!isTRUE(filter_jumps)) {
    filter_graph = po("filter_jumps") %>>% filter_graph
  }

  # Check factors and build pipeline
  if (!("factor" %in% learner$feature_types) & "factor" %in% unique(task$feature_types$type)) {
    cat("Factor encoding for learner", learner$id, "\n")
    learner = filter_graph %>>%
      po("removeconstants") %>>%
      po("collapsefactors", no_collapse_above_prevalence = 0.01) %>>%
      po("encode", method = "one-hot") %>>%
      po("learner", learner) |>
      as_learner()
  } else {
    learner = filter_graph %>>%
      po("learner", learner) |>
      as_learner()
  }
  # plot(learner)

  # Check search space
  sp = as.data.table(learner$param_set)
  if (any((search_space$ids() %in% sp$id) == FALSE)) {
    print(sp$id)
    stop("Check search space!")
  }

  # Use inner resampling for validation (80/20 split)
  inner_rsmp = rsmp("holdout", ratio = 0.8)

  # Create autotuner
  at = auto_tuner(
    learner = learner,
    resampling = inner_rsmp,
    measure = msr("regr.mse"),
    search_space = search_space,
    terminator = if (tuner$label == "Hyperband") trm("none") else trm("evals", n_evals = n_evals),
    tuner = tuner
  )
  set_threads(at, n = threads)
  
  return(at)
}

# Test if training work with/without jumps
if (interactive()) {
  tsk_ = task$clone()
  tsk_$filter(1:20000)
  tsk_$data(cols = "jump")[jump == 1]
  po = PipeOpFilterJumps$new()
  r = po$train(list(tsk_))
  print(tsk_$nrow)
  print(r$output$nrow)
}

# Parameters
n_evals = 5
threads = 4

# Filter search space (reused across models)
# Only tune filter selection, keep filter.frac fixed to compare methods fairly
filter_ps = ps(
  filter_branch.selection = p_fct(levels = c("jmim", "mrmr", "cmim", "importance"))
)

# Random forest
at_rf = create_autotuner(
  learner      = lrn("regr.ranger", id = "ranger"),
  search_space = c(ps(
    ranger.max.depth  = p_int(1, 15),
    ranger.replace    = p_lgl(),
    ranger.mtry.ratio = p_dbl(0.3, 1),
    ranger.splitrule  = p_fct(levels = c("variance", "extratrees")),
    ranger.num.trees  = p_int(10, 2000, tags = "budget")
  ), filter_ps)
)
at_rf_adj = create_autotuner(
  learner      = lrn("regr.ranger", id = "ranger_jump"),
  search_space = c(ps(
    ranger_jump.max.depth  = p_int(1, 15),
    ranger_jump.replace    = p_lgl(),
    ranger_jump.mtry.ratio = p_dbl(0.3, 1),
    ranger_jump.splitrule  = p_fct(levels = c("variance", "extratrees")),
    ranger_jump.num.trees  = p_int(10, 2000, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# XGBOOST
at_xgboost = create_autotuner(
  learner      = lrn("regr.xgboost", id = "xgboost"),
  search_space = c(ps(
    xgboost.alpha     = p_dbl(0.001, 100, logscale = TRUE),
    xgboost.max_depth = p_int(1, 20),
    xgboost.eta       = p_dbl(0.0001, 1, logscale = TRUE),
    xgboost.subsample = p_dbl(0.1, 1),
    xgboost.nrounds   = p_int(30, 5000, tags = "budget")
  ), filter_ps)
)
at_xgboost_adj = create_autotuner(
  learner      = lrn("regr.xgboost", id = "xgboost_jump"),
  search_space = c(ps(
    xgboost_jump.alpha     = p_dbl(0.001, 100, logscale = TRUE),
    xgboost_jump.max_depth = p_int(1, 20),
    xgboost_jump.eta       = p_dbl(0.0001, 1, logscale = TRUE),
    xgboost_jump.subsample = p_dbl(0.1, 1),
    xgboost_jump.nrounds   = p_int(30, 5000, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# NNET
at_nnet = create_autotuner(
  learner      = lrn("regr.nnet", id = "nnet", MaxNWts = 50000),
  search_space = c(ps(
    nnet.size  = p_int(lower = 2, upper = 15),
    nnet.decay = p_dbl(lower = 0.0001, upper = 0.1),
    nnet.maxit = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps)
)
at_nnet_adj = create_autotuner(
  learner      = lrn("regr.nnet", id = "nnet_jump", MaxNWts = 50000),
  search_space = c(ps(
    nnet_jump.size  = p_int(lower = 2, upper = 15),
    nnet_jump.decay = p_dbl(lower = 0.0001, upper = 0.1),
    nnet_jump.maxit = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# BART
at_bart = create_autotuner(
  learner      = lrn("regr.bart", id = "bart", sigest = 1),
  search_space = c(ps(
    bart.k      = p_dbl(lower = 1, upper = 8),
    bart.numcut = p_int(lower = 30, upper = 200),
    bart.ntree  = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps)
)
at_bart_adj = create_autotuner(
  learner      = lrn("regr.bart", id = "bart_jump", sigest = 1),
  search_space = c(ps(
    bart_jump.k      = p_dbl(lower = 1, upper = 8),
    bart_jump.numcut = p_int(lower = 30, upper = 200),
    bart_jump.ntree  = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# NN
mlp_graph = po("torch_ingress_num") %>>%
  po("nn_linear", out_features = 20) %>>%
  po("nn_relu") %>>%
  po("nn_head") %>>%
  po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
  po("torch_callbacks", callbacks = t_clbk("history")) %>>%
  po("torch_model_regr", batch_size = 16, epochs = 50, device = "cpu")
at_nn = create_autotuner(
  learner      = mlp_graph,
  search_space = c(ps(
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_model_regr.batch_size = p_int(lower = 16, upper = 256, tags = "tune"),
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_optimizer.lr = p_dbl(lower = 1e-5, upper = 1e-1, logscale = TRUE, tags = "tune"),
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_model_regr.epochs = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps)
)
at_nn_adj = create_autotuner(
  learner      = mlp_graph,
  search_space = c(ps(
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_model_regr.batch_size = p_int(lower = 16, upper = 256, tags = "tune"),
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_optimizer.lr = p_dbl(lower = 1e-5, upper = 1e-1, logscale = TRUE, tags = "tune"),
    torch_ingress_num.nn_linear.nn_relu.nn_head.torch_loss.torch_optimizer.torch_callbacks.torch_model_regr.torch_model_regr.epochs = p_int(lower = 50, upper = 500, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# earth
at_earth = create_autotuner(
  learner      = lrn("regr.earth", id = "earth"),
  search_space = c(ps(
    earth.degree  = p_int(lower = 1, upper = 3),
    earth.penalty = p_dbl(lower = 1, upper = 5),
    earth.nprune  = p_int(lower = 10, upper = 100),
    earth.pmethod = p_fct(levels = c("backward", "none", "exhaustive", "forward")),
    earth.nk      = p_int(lower = 50, upper = 300, tags = "budget")
  ), filter_ps)
)
at_earth_adj = create_autotuner(
  learner      = lrn("regr.earth", id = "earth_jump"),
  search_space = c(ps(
    earth_jump.degree  = p_int(lower = 1, upper = 3),
    earth_jump.penalty = p_dbl(lower = 1, upper = 5),
    earth_jump.nprune  = p_int(lower = 10, upper = 100),
    earth_jump.pmethod = p_fct(levels = c("backward", "none", "exhaustive", "forward")),
    earth_jump.nk      = p_int(lower = 50, upper = 300, tags = "budget")
  ), filter_ps),
  filter_jumps = FALSE
)

# GBM (Gradient Boosting Machine)
at_gbm = create_autotuner(
  learner      = lrn("regr.gbm", id = "gbm"),
  search_space = c(ps(
    gbm.interaction.depth = p_int(lower = 1, upper = 10),
    gbm.shrinkage         = p_dbl(lower = 0.001, upper = 0.1, logscale = TRUE),
    gbm.bag.fraction      = p_dbl(lower = 0.5, upper = 1),
    gbm.n.minobsinnode    = p_int(lower = 5, upper = 30),
    gbm.n.trees           = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps)
)
at_gbm_adj = create_autotuner(
  learner      = lrn("regr.gbm", id = "gbm_jumo"),
  search_space = c(ps(
    gbm_jumo.interaction.depth = p_int(lower = 1, upper = 10),
    gbm_jumo.shrinkage         = p_dbl(lower = 0.001, upper = 0.1, logscale = TRUE),
    gbm_jumo.bag.fraction      = p_dbl(lower = 0.5, upper = 1),
    gbm_jumo.n.minobsinnode    = p_int(lower = 5, upper = 30),
    gbm_jumo.n.trees           = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps),
  filter_jumps = TRUE
)

# LightGBM - Fast, efficient gradient boosting
at_lightgbm = create_autotuner(
  learner      = lrn("regr.lightgbm", id = "lightgbm"),
  search_space = c(ps(
    lightgbm.num_leaves         = p_int(lower = 7, upper = 127),
    lightgbm.learning_rate      = p_dbl(lower = 0.001, upper = 0.3, logscale = TRUE),
    lightgbm.feature_fraction   = p_dbl(lower = 0.5, upper = 1),
    lightgbm.bagging_fraction   = p_dbl(lower = 0.5, upper = 1),
    lightgbm.min_data_in_leaf   = p_int(lower = 5, upper = 50),
    lightgbm.num_iterations     = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps)
)
at_lightgbm_adj = create_autotuner(
  learner      = lrn("regr.lightgbm", id = "lightgbm_jump"),
  search_space = c(ps(
    lightgbm_jump.num_leaves         = p_int(lower = 7, upper = 127),
    lightgbm_jump.learning_rate      = p_dbl(lower = 0.001, upper = 0.3, logscale = TRUE),
    lightgbm_jump.feature_fraction   = p_dbl(lower = 0.5, upper = 1),
    lightgbm_jump.bagging_fraction   = p_dbl(lower = 0.5, upper = 1),
    lightgbm_jump.min_data_in_leaf   = p_int(lower = 5, upper = 50),
    lightgbm_jump.num_iterations     = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps),
  filter_jumps = FALSE
)

# Cubist - Rule-based model with linear models
at_cubist = create_autotuner(
  learner      = lrn("regr.cubist", id = "cubist"),
  search_space = c(ps(
    cubist.committees = p_int(lower = 1, upper = 50, tags = "budget"),
    cubist.neighbors  = p_int(lower = 0, upper = 7),
    cubist.unbiased   = p_lgl(),
    cubist.rules      = p_int(lower = 10, upper = 70),
    cubist.extrapolation = p_dbl(lower = 0, upper = 70)
  ), filter_ps)
)
at_cubist_adj = create_autotuner(
  learner      = lrn("regr.cubist", id = "cubist_jump"),
  search_space = c(ps(
    cubist_jump.committees = p_int(lower = 1, upper = 50, tags = "budget"),
    cubist_jump.neighbors  = p_int(lower = 0, upper = 7),
    cubist_jump.unbiased   = p_lgl(),
    cubist_jump.rules      = p_int(lower = 10, upper = 70),
    cubist_jump.extrapolation = p_dbl(lower = 0, upper = 70)
  ), filter_ps),
  filter_jumps = FALSE
)

# CatBoost - handles factors natively!
at_catboost = create_autotuner(
  learner = lrn("regr.catboost", id = "catboost"),
  search_space = c(ps(
    catboost.depth          = p_int(lower = 4, upper = 10),
    catboost.learning_rate  = p_dbl(lower = 0.01, upper = 0.3, logscale = TRUE),
    catboost.l2_leaf_reg    = p_dbl(lower = 1, upper = 10),
    catboost.bagging_temperature = p_dbl(lower = 0, upper = 1),
    catboost.iterations     = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps)
)
at_catboost_adj = create_autotuner(
  learner = lrn("regr.catboost", id = "catboost_jump"),
  search_space = c(ps(
    catboost_jump.depth          = p_int(lower = 4, upper = 10),
    catboost_jump.learning_rate  = p_dbl(lower = 0.01, upper = 0.3, logscale = TRUE),
    catboost_jump.l2_leaf_reg    = p_dbl(lower = 1, upper = 10),
    catboost_jump.bagging_temperature = p_dbl(lower = 0, upper = 1),
    catboost_jump.iterations     = p_int(lower = 100, upper = 3000, tags = "budget")
  ), filter_ps),
  filter_jumps = FALSE
)

# Oblique random survival forest
at_aorsf = create_autotuner(
  learner = lrn("regr.aorsf", id = "aorsf"),
  search_space = c(ps(
    aorsf.n_tree = p_int(lower = 100, upper = 2000, tags = "budget"),
    aorsf.mtry = p_int(lower = 1, upper = 20),
    aorsf.leaf_min_obs = p_int(lower = 5, upper = 50),
    aorsf.split_min_obs = p_int(lower = 10, upper = 100)
  ), filter_ps)
)

# GAM
# at_gamboost = create_autotuner(
#   learner = lrn("regr.gamboost", id = "gamboost"),
#   search_space = ps(
#     mstop  = p_int(lower = 50, upper = 1000, tags = "budget"),  # BUDGET parameter!
#     nu     = p_dbl(lower = 0.01, upper = 0.5)                   # Learning rate
#   )
# )

# Combine autotuners
autotuners = list(
  at_rf, at_xgboost, at_nnet, at_bart, at_nn, at_earth, at_gbm, at_lightgbm, at_cubist, 
  at_catboost, at_aorsf
)
autotuners_jump = list(
    at_rf_adj, at_xgboost_adj, at_nnet_adj, at_bart_adj, at_nn_adj, at_earth_adj,
    at_gbm_adj, at_lightgbm_adj, at_cubist_adj, at_catboost_adj
  )
if (isTRUE(JUMP)) {
  autotuners = c(autotuners, autotuners_jump)
}

# Mlr3 design
design = benchmark_grid(
  tasks = task,
  learners = autotuners, 
  resamplings = rsmp("gap_cv", initial_window = train_size_years_init, horizon = 1, gap = 0, step = 1, rolling = FALSE)
)

# Checks
if (interactive()) {
  design$resampling[[1]]$iters
}

# # Benchmark
# bmr = benchmark(design)

# create registry
packages = c("data.table", "paradox", "mlr3", "mlr3pipelines",
             "mlr3tuning", "mlr3misc", "future", "future.apply",
             "mlr3extralearners")
if (interactive()) {
  if (dir.exists("./experiments_test")) {
    fs::dir_delete("./experiments_test")
  }
  reg = makeExperimentRegistry(
    file.dir = "./experiments_test",
    seed = 1,
    packages = packages
  )
} else {
  reg = makeExperimentRegistry(
    file.dir = "./experiments_fi",
    seed = 1,
    packages = packages
  )
}

# populate registry with problems and algorithms to form the jobs
batchmark(design, reg = reg, store_models = TRUE)

# save registry
saveRegistry(reg = reg)

# Test 2 ids (sentiment and non sentiment)
if (interactive()) {
  # cluter function template
  cf = makeClusterFunctionsSocket(ncpus = 2)
  reg$cluster.functions = cf
  saveRegistry(reg = reg)
  reg = loadRegistry(file.dir = "./experiments_test",
                     work.dir = "./experiments_test",
                     writeable = TRUE)

  # set resources
  resources = list(ncpus = 2)

  # get unsubmited ids
  findNotDone(reg = reg)

  # submit jobs
  ids = 2
  submitJobs(ids = ids, resources = resources, reg = reg)

  # Read results
  reg = loadRegistry("./experiments_test")
  res = reduceResultsBatchmark(ids = 2, reg = reg)
  res$resample_results
  res$resample_results$resample_result[[1]]
  res$aggregate()
}

# create sh file
reg_folder = if (interactive()) "experiments_test" else "experiments_fi"
reg = loadRegistry(reg_folder, reg_folder, writeable = TRUE)
ids = findNotDone(reg = reg)
sh_file = sprintf("
#!/bin/bash

#PBS -N HFFZFI
#PBS -l ncpus=4
#PBS -l mem=46GB
#PBS -l walltime=90:00:00
#PBS -J 1-%d
#PBS -o experiments_fi/logs
#PBS -j oe

cd ${PBS_O_WORKDIR}
apptainer run image.sif run_job.R 0 experiments_fi
", nrow(ids))
sh_file_name = "padobran_fi.sh"
file.create(sh_file_name)
writeLines(sh_file, sh_file_name)


# FEATURE IMPORTANCE -----------------------------
if (interactive()) {
  library(mlr3filters)
  library(ggplot2)
  
  # Create resampling (same as in benchmark)
  r = rsmp("gap_cv", 
           initial_window = train_size_years_init, 
           horizon = 1, 
           gap = 0, 
           step = 1, 
           rolling = FALSE)
  r$instantiate(task)
  
  all_selected_features = list()
  all_scores = list()
  
  for (fold in seq_len(r$iters)) {
    # fold = 1
    cat("\nProcessing fold", fold, "/", r$iters, "\n")
    
    # Get training set for this fold
    train_ids = r$train_set(fold)
    task_fold = task$clone()
    task_fold$filter(train_ids)
    
    # Calculate JMIM on training data
    filter_jmim = flt("jmim")
    filter_jmim$calculate(task_fold, nfeat = ceiling(task_fold$n_features * filter_frac))
    
    # Get feature scores
    feature_scores = as.data.table(filter_jmim)
    feature_scores = na.omit(feature_scores)
        
    # Get selected features (top N based on filter_frac)
    selected_features = feature_scores$feature
    
    # Store results
    all_selected_features[[fold]] = data.table(
      fold = fold,
      feature = selected_features,
      score = feature_scores$score
    )
    
    all_scores[[fold]] = feature_scores[, .(fold = fold, feature, score)]
  }
  
  # Combine all folds
  selected_features_dt = rbindlist(all_selected_features)

  # Analyze feature stability across folds
  feature_frequency = selected_features_dt[, .N, by = feature][order(-N)]
  feature_frequency[, feature_best := gsub("_day|_hour", "", feature)]
  feature_frequency[, sum(N), by = feature_best][order(-V1)]
  feature_frequency[grepl("ind48", feature_best)]

  # Save results
  saveRDS(list(
    selected_by_fold = selected_features_dt,
    all_scores = all_scores_dt,
    frequency = feature_frequency
  ), "data/jmim_features_by_fold.rds")
  
  fwrite(selected_features_dt, "data/jmim_selected_by_fold.csv")
  fwrite(feature_frequency, "data/jmim_feature_frequency.csv")
  
  # Visualize feature stability
  p1 = ggplot(feature_frequency[1:30], 
              aes(x = reorder(feature, frequency), y = frequency)) +
    geom_col() +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
    coord_flip() +
    labs(title = "Top 30 Features by Selection Frequency", 
         x = "Feature", 
         y = "Proportion of Folds Selected") +
    theme_minimal()
  
  # Visualize score evolution over time
  top_features = feature_frequency[1:10, feature]
  scores_top = all_scores_dt[feature %in% top_features]
  
  p2 = ggplot(scores_top, aes(x = fold, y = score, color = feature)) +
    geom_line() +
    geom_point() +
    labs(title = "JMIM Score Evolution for Top 10 Features",
         x = "Fold (Time)", 
         y = "JMIM Score") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p1)
  print(p2)
}