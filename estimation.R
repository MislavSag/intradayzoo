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


# Import data
factors = fread("data/factor_returns.csv")

# Prepare data
head(colnames(factors)); tail(colnames(factors))
id_columns = c("datetime", "jump")
factors[, year := data.table::year(datetime)]

# Define tasks
task  = as_task_regr(factors[, .SD, .SDcols = -c("targetc")], "target", "factorszoo")
taskc = as_task_regr(factors[, .SD, .SDcols = -c("target")], "targetc", "factorszooc")

# set roles for id columns
task$set_col_roles("year", "group")
task$col_roles$feature = setdiff(task$col_roles$feature, id_columns)
taskc$set_col_roles("year", "group")
taskc$col_roles$feature = setdiff(taskc$col_roles$feature, id_columns)

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
create_autotuner = function(learner, search_space, n_evals = 20) {
  # Use random search for efficiency
  tuner = tnr("random_search")
  
  # Use inner resampling for validation (80/20 split)
  # This is the validation set mentioned in the paper
  inner_rsmp = rsmp("holdout", ratio = 0.8)
  
  # Create autotuner
  at = auto_tuner(
    learner = learner,
    resampling = inner_rsmp,
    measure = msr("regr.mse"),
    search_space = search_space,
    # terminator = trm("evals", n_evals = n_evals),
    term_evals = n_evals,
    tuner = tuner
  )
  
  return(at)
}

# Random forest
at_rf = create_autotuner(
  learner      = lrn("regr.ranger", id = "ranger"),
  search_space = ps(
    max.depth  = p_int(1, 15),
    replace    = p_lgl(),
    mtry.ratio = p_dbl(0.3, 1),
    num.trees  = p_int(10, 2000),
    splitrule  = p_fct(levels = c("variance", "extratrees"))
  ),
  n_evals = 20
)

# XGBOOST
at_xgboost = create_autotuner(
  learner      = lrn("regr.xgboost", id = "xgboost"),
  search_space = ps(
    alpha     = p_dbl(0.001, 100, logscale = TRUE),
    max_depth = p_int(1, 20),
    eta       = p_dbl(0.0001, 1, logscale = TRUE),
    nrounds   = p_int(1, 5000),
    subsample = p_dbl(0.1, 1)
  ),
  n_evals = 20
)

# NNET
at_nnet = create_autotuner(
  learner      = lrn("regr.nnet", id = "nnet"),
  search_space = ps(
    size  = p_int(lower = 2, upper = 15),
    decay = p_dbl(lower = 0.0001, upper = 0.1),
    maxit = p_int(lower = 50, upper = 500)
    
  ),
  n_evals = 20
)

# Mlr3 design
autotuners = list(at_rf, at_xgboost, at_nnet)
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
    file.dir = "./experiments",
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
reg_folder = if (interactive()) "experiments_test" else "experiments"
reg = loadRegistry(reg_folder, reg_folder, writeable = TRUE)
ids = findNotDone(reg = reg)
sh_file = sprintf("
#!/bin/bash

#PBS -N HFFZ
#PBS -l ncpus=4
#PBS -l mem=10GB
#PBS -l walltime=90:00:00
#PBS -J 1-%d
#PBS -o experiments/logs
#PBS -j oe

cd ${PBS_O_WORKDIR}
apptainer run image.sif run_job.R 0
", nrow(ids))
sh_file_name = "padobran.sh"
file.create(sh_file_name)
writeLines(sh_file, sh_file_name)

# apptainer run image_estimation.sif h4_run_job.R 0



# ARCHIVE AND HELP
# add my pipes to mlr dictionary
# pipes = as.data.table(mlr_pipeops)
# pipes[grepl("linex", key)]
# mlr_filters$add("gausscov_f1st", finautoml::FilterGausscovF1st)
# # mlr_filters$add("gausscov_f3st", FilterGausscovF3st) # not found error
# mlr_measures$add("linex", finautoml::Linex)
# mlr_measures$add("adjloss2", finautoml::AdjLoss2)
# mlr_measures$add("portfolio_ret", PortfolioRet)

# # test MLP learner
# mlp_graph = po("torch_ingress_num") %>>%
#   po("nn_linear", out_features = 20) %>>%
#   po("nn_relu") %>>%
#   po("nn_head") %>>%
#   po("torch_loss", loss = t_loss("mse")) %>>%
#   po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
#   po("torch_callbacks", callbacks = t_clbk("history")) %>>%
#   po("torch_model_regr", batch_size = 16, epochs = 50, device = "cpu")

# # cretate learners graph node
# learners_l = list(
#   ranger  = lrn("regr.ranger", id = "ranger"),
#   xgboost = lrn("regr.xgboost", id = "xgboost"),
#   # bart    = lrn("regr.bart", id = "bart", sigest = 1),
#   nnet    = lrn("regr.nnet", id = "nnet", MaxNWts = 50000),
#   mlp     = mlp_graph
# )

# # create regression average of all learners
# choices = c("ranger", "xgboost", "bart", "nnet", "mlp")
# learners = po("branch", choices) %>>%
#   gunion(learners_l) %>>%
#   po("unbranch")

# # Hyperparameters
# search_space = ps(
#   # # scaling
#   # scale_branch.selection = p_fct(levels = c("uniformization", "scale")),
#   # histbin_sentiment.breaks = p_fct(
#   #   levels = c("5", "10", "20", "50"),
#   #   trafo = function(x, param_set) {
#   #     switch(
#   #       x,
#   #       "5" = 5,
#   #       "10" = 10,
#   #       "20" = 20,
#   #       "50" = 50
#   #     )
#   #   }
#   # ),
#   # models
#   choose_learners.selection = p_fct(levels = choices),
#   mlp.torch_model_regr.batch_size = p_int(lower = 8, upper = 128, tags = "tune", depends = choose_learners.selection == "mlp"),
#   # mlp.torch_model_regr.patience = p_int(lower = 0, upper = 10, tags = "tune"),
#   mlp.torch_optimizer.lr = p_dbl(lower = 1e-4, upper = 1e-1, logscale = TRUE, tags = "tune", depends = choose_learners.selection == "mlp"),
#   mlp.torch_model_regr.epochs = p_int(lower = 20, upper = 200, tags = "tune", depends = choose_learners.selection == "mlp"),
#   tabnet.tabnet.epochs = p_int(lower = 20, upper = 200, tags = "tune", depends = choose_learners.selection == "tabnet"),
#   tabnet.tabnet.virtual_batch_size = p_int(lower = 16, upper = 512, tags = "tune", depends = choose_learners.selection == "tabnet"),
#   tabnet.tabnet.learn_rate = p_dbl(lower = 1e-4, upper = 1e-1, logscale = TRUE, tags = "tune", depends = choose_learners.selection == "tabnet"),
#   tabnet.tabnet.num_steps = p_int(lower = 2, upper = 10, tags = "tune", depends = choose_learners.selection == "tabnet"),
#   tabnet.tabnet.decision_width = p_int(lower = 8, upper = 64, tags = "tune", depends = choose_learners.selection == "tabnet"),
#   bart.bart.k      = p_dbl(lower = 1, upper = 10, depends = choose_learners.selection == "bart"),
#   bart.bart.numcut = p_int(lower = 10, upper = 200, depends = choose_learners.selection == "bart"),
#   bart.bart.ntree  = p_int(lower = 50, upper = 500, depends = choose_learners.selection == "bart")
# )

# # Learners without hyperparameter tuning
# fixed_learners = list(
#   ols = linear_learners$ols,
#   pcr = linear_learners$pcr
# )
