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


# Import data
factors = fread("data/factor_returns_overnight.csv")

# Prepare data
factors[, c("date", "time"):= NULL]
head(colnames(factors)); tail(colnames(factors))
id_columns = c("datetime", "datetime_overnight")
factors[, year := data.table::year(datetime)]
factors[, .(datetime, datetime_overnight, target)]

# Remove missing values
dim(factors)
factors = na.omit(factors)
dim(factors)

# Define tasks
task  = as_task_regr(factors, "target", "factorszoo")

# set roles for id columns
task$set_col_roles("year", "group")
task$col_roles$feature = setdiff(task$col_roles$feature, id_columns)

# Cross validation resampling parameters
FIRST_YEAR = 2005
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
  include_jumps = TRUE) {
  
  # Use inner resampling for validation (80/20 split)
  # This is the validation set mentioned in the paper
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

# Parameters
n_evals = 5
threads = 4

# Random forest
at_rf = create_autotuner(
  learner      = lrn("regr.ranger", id = "ranger"),
  search_space = ps(
    max.depth  = p_int(1, 15),
    replace    = p_lgl(),
    mtry.ratio = p_dbl(0.3, 1),
    splitrule  = p_fct(levels = c("variance", "extratrees")),
    # num.trees  = p_int(10, 2000)
    num.trees  = p_int(10, 2000, tags = "budget")  # Budget parameter
  )
)

# XGBOOST
at_xgboost = create_autotuner(
  learner      = lrn("regr.xgboost", id = "xgboost"),
  search_space = ps(
    alpha     = p_dbl(0.001, 100, logscale = TRUE),
    max_depth = p_int(1, 20),
    eta       = p_dbl(0.0001, 1, logscale = TRUE),
    subsample = p_dbl(0.1, 1),
    # nrounds   = p_int(1, 5000),
    nrounds   = p_int(30, 5000, tags = "budget")  # Budget parameter
  )
)

# NNET
at_nnet = create_autotuner(
  learner      = lrn("regr.nnet", id = "nnet", MaxNWts = 50000),
  search_space = ps(
    size  = p_int(lower = 2, upper = 15),
    decay = p_dbl(lower = 0.0001, upper = 0.1),
    # maxit = p_int(lower = 50, upper = 500)
    maxit = p_int(lower = 50, upper = 500, tags = "budget")  # Budget parameter
  )
)

# BART
at_bart = create_autotuner(
  learner      = lrn("regr.bart", id = "bart", sigest = 1),
  search_space = ps(
    k      = p_dbl(lower = 1, upper = 8),
    numcut = p_int(lower = 30, upper = 200),
    ntree  = p_int(lower = 50, upper = 500, tags = "budget")  # Budget parameter
  )
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
  search_space = ps(
    torch_model_regr.batch_size = p_int(lower = 16, upper = 256, tags = "tune"), # Batch size
    torch_optimizer.lr = p_dbl(lower = 1e-5, upper = 1e-1, logscale = TRUE, tags = "tune"), # Learning rate
    torch_model_regr.epochs = p_int(lower = 50, upper = 500, tags = "budget")   # BUDGET: training epochs
  )
)

# earth
at_earth = create_autotuner(
  learner      = lrn("regr.earth", id = "earth"),
  search_space = ps(
    degree  = p_int(lower = 1, upper = 3),                      # Max interaction degree
    penalty = p_dbl(lower = 1, upper = 5),                      # GCV penalty per knot
    nprune  = p_int(lower = 10, upper = 100),                   # Max terms after pruning
    pmethod = p_fct(levels = c("backward", "none", "exhaustive", "forward")), # Pruning method
    nk      = p_int(lower = 50, upper = 300, tags = "budget")   # BUDGET: max terms before pruning
  ),
 )

# 

# GBM (Gradient Boosting Machine)
at_gbm = create_autotuner(
  learner      = lrn("regr.gbm", id = "gbm"),
  search_space = ps(
    interaction.depth = p_int(lower = 1, upper = 10),        # Tree depth
    shrinkage         = p_dbl(lower = 0.001, upper = 0.1, logscale = TRUE), # Learning rate
    bag.fraction      = p_dbl(lower = 0.5, upper = 1),       # Subsampling fraction
    n.minobsinnode    = p_int(lower = 5, upper = 30),        # Min observations in terminal nodes
    n.trees           = p_int(lower = 100, upper = 3000, tags = "budget")  # BUDGET: number of trees
  )
)

# Mlr3 design
autotuners = list(
  at_rf, at_xgboost, at_nnet, at_bart, at_nn, at_earth,
  at_rf_adj, at_xgboost_adj, at_nnet_adj, at_bart_adj, at_nn_adj, at_earth_adj
)
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
batchmark(design, reg = reg, store_models = FALSE)

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
#PBS -l mem=22GB
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
