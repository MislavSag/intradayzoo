library(data.table)
library(mlr3batchmark)
library(batchtools)
library(httr)
library(arrow)
library(lubridate)
library(PerformanceAnalytics)
library(AzureStor)
library(ggplot2)


# Import registry
reg = loadRegistry("./experiments_panel", "./experiments_panel", writeable = TRUE)
getStatus(reg = reg)
job_table = getJobTable(reg = reg)
job_table = unwrap(job_table)
job_table = job_table[,
  .(job.id, learner_id, task_id, resampling_id, repl)
]
findErrors(reg = reg)

# Import results
ids_ = findDone(reg = reg)
# remove_ids = 17*c(1:6)
# ids_ = ids_[job.id %notin% remove_ids]
bmr = reduceResultsBatchmark(ids_, store_backends = TRUE, reg = reg)

# Aggregate results
bmr$aggregate() |>
  _[order(regr.mse)]

# Detailed results with all measures
bmr_scores = bmr$score(measures = list(msr("regr.mse"), msr("regr.mae"), msr("regr.rmse")))
print(head(bmr_scores))

# Extract predictions
bmr_with_preds = bmr$score(predictions = TRUE)

# Predictions data.table
predictions_list = list()
for (i in seq_len(nrow(bmr_with_preds))) {
  # i = 1
  # Get prediction
  pred = bmr_with_preds$prediction_test[[i]]
  pred_dt = as.data.table(pred)
  
  # Get metadata
  learner_id = bmr_with_preds$learner_id[i]
  iteration = bmr_with_preds$iteration[i]
  task = bmr_with_preds$task[[i]]
  
  # Extract datetime from task backend using row_ids
  row_ids = pred_dt$row_ids
  datetime_data = task$backend$data(rows = row_ids, cols = c("symbol", "date"))
  datetime_data[, date := with_tz(datetime_data[, date], tzone = "America/New_York")]
  
  # Combine into data.table
  result_dt = data.table(
    symbol = datetime_data$symbol,
    datetime = datetime_data$date,
    row_ids = pred_dt$row_ids,
    learner_id = learner_id,
    iteration = iteration,
    truth = pred_dt$truth,
    response = pred_dt$response
  )
  
  predictions_list[[i]] = result_dt
}
predictions_dt = rbindlist(predictions_list)

# Plot returns
predictions_dt[, unique(symbol)]
s = "JNJ"
predictions_dt[symbol == s & grepl("nnet", learner_id), .(datetime, truth)] |>
  as.xts.data.table() |>
  charts.PerformanceSummary()

# Add ensamble
back = copy(predictions_dt)
cols = back[, unique(learner_id)]
back[, learner_id := gsub("removeconstants.collapsefactors.encode.", "", learner_id)]
back[, learner_id := gsub("\\..*", "", learner_id)]
cols = back[, unique(learner_id)]
back = dcast(back, symbol + datetime + truth ~ learner_id, value.var = "response") |>
  _[, let(
    ensamble_mean   = matrixStats::rowMeans2(as.matrix(.SD), na.rm = TRUE),
    ensamble_median   = matrixStats::rowMedians(as.matrix(.SD), na.rm = TRUE),
    ensamble_max   = matrixStats::rowMaxs(as.matrix(.SD), na.rm = TRUE)
  ), .SDcols = cols] |>
  melt(
    id.vars = c("symbol", "datetime", "truth"), 
    variable.name = "learner_id",
    value.name = "response",
    variable.factor = FALSE
  )

# Approximat trading costs - Conservative estimate (market orders, small size):
round_trip_cost = 0.00002 + 0.00002 + 0.00003  # spread both sides + fees = 0.00007 (0.007%)
one_way_cost = 0.00002 + 0.000015 

# S-sign trading rule
back[, .(mean(response, na.rm = TRUE), median(response, na.rm = TRUE), mean(truth, na.rm = TRUE), median(truth, na.rm = TRUE))]
half_spread = one_way_cost / 2
back[, weight := NA_integer_]
back[response > half_spread, weight := 1L]
back[response < -half_spread, weight := -1L]
back[is.na(weight)]
setorder(back, symbol, learner_id, datetime)
back[, weight := nafill(weight, type = "locf"), by = .(symbol, learner_id)]
back[, ret_truth := truth * weight]
back[, ret_truth_lead := shift(truth, 1L, type = "lead") * weight]

# remove na weight
back = na.omit(back, cols = "response")

# Calculate actual trading costs per period
back[, weight_change := abs(c(NA, diff(ifelse(weight == -1, 0, weight)))), by = .(learner_id)]
back[, trading_cost := weight_change * one_way_cost]
back[is.na(trading_cost), trading_cost := 0]

# Net returns after costs
back[, ret_truth_net := ret_truth - trading_cost]
back[, ret_truth_lead_net := ret_truth_lead - trading_cost]

# Summary: total costs per learner
back[, .(
  total_trades = sum(weight_change, na.rm = TRUE),
  total_cost_bps = sum(trading_cost, na.rm = TRUE) * 10000,
  avg_cost_per_period_bps = mean(trading_cost, na.rm = TRUE) * 10000
), by = .(symbol, learner_id)]

# Predicttions to wide format
back[, .N, by = learner_id]
models_to_keep = back[, .N > 900000, by = learner_id][V1 == TRUE, learner_id]
port = back[, .(symbol, datetime, truth, ret_truth, ret_truth_net, learner_id)] |>
  dcast(symbol + datetime + truth ~ learner_id, value.var = "ret_truth") |>
  _[, .SD, .SDcols = c("symbol", "datetime", "truth", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(symbol, datetime = as.Date(datetime))]
port_net = back[, .(symbol, datetime, truth, ret_truth, ret_truth_net, learner_id)] |>
  dcast(symbol + datetime + truth ~ learner_id, value.var = "ret_truth_net") |>
  _[, .SD, .SDcols = c("symbol", "datetime", "truth", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(symbol, date = as.Date(datetime))]
port_lead = back[, .(symbol, datetime, truth, ret_truth_lead, ret_truth_net, learner_id)] |>
  dcast(symbol + datetime + truth ~ learner_id, value.var = "ret_truth_lead") |>
  _[, .SD, .SDcols = c("symbol", "datetime", "truth", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(symbol, date = as.Date(datetime))]


# INDIVIDUAL PERFORMANCE ----------------------------------------
# Performance
back[, unique(symbol)]
symbol_ = "MSFT"
melt(as.data.table(SharpeRatio.annualized(as.xts.data.table(port[symbol == symbol_, .SD, .SDcols = -"symbol"]), scale = 252)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(as.xts.data.table(port_net[symbol == symbol_, .SD, .SDcols = -"symbol"]), scale = 252)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(as.xts.data.table(port_lead[symbol == symbol_, .SD, .SDcols = -"symbol"]), scale = 252)))[order(-value)]

# Biggest decrease gorss / net
x = melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252)))
y = melt(as.data.table(SharpeRatio.annualized(port_spy_net, scale = 252)))  # Net of actual costs
cbind(x[, 1], x[, 2] - y[, 2])[order(value)]

# Capital curvesback[symbol == "AAPL" & learner_id == "bart", .(datetime, truth)] |>
charts.PerformanceSummary(port[symbol == symbol_, .SD, .SDcols = -"symbol"])
charts.PerformanceSummary(port_net[symbol == symbol_, .SD, .SDcols = -"symbol"])  # Net of actual costs


# AGGREAGETE PERFORMANCE -----------------------------------------
# Best models
sr = port[, as.data.table(SharpeRatio.annualized(as.xts.data.table(.SD)), scale = 252), by = symbol]
sr[, lapply(.SD, function(x) sum(x - truth, na.rm = TRUE)), .SDcols = 2:ncol(sr)] |>
  melt() |>
  _[order(value)]

# SR ratios acroos models
port[, .(symbol, datetime, truth, ensamble_median)] |>
  _[, as.data.table(SharpeRatio.annualized(as.xts.data.table(.SD)), scale = 252), by = symbol] |>
  melt(is.vars = "symbol") |>
  ggplot(aes(x = symbol, y = value, color = variable, group = variable)) +
    geom_point(size = 3) +
    geom_line() +
    theme_minimal()


# CROSS SECTION PERFORMANCE --------------------------------------
# Equal weights SR
ew = back[learner_id == "ensamble_median", .(symbol, datetime, truth)]
ew = ew[, sum(truth * (1/length(truth)), na.rm = TRUE), by = datetime]
ew = ew[, lapply(.SD, function(x) Return.cumulative(x)), by = .(datetime = as.Date(datetime))]
setorder(ew, datetime)
SharpeRatio.annualized(ew, scale = 252)

# Filter learner_id and colums
back_cs = back[learner_id == "ensamble_median", .(symbol, datetime, truth, response)]
setorder(back_cs, datetime, -response)

# Check same number of predictions
same_predictions = back_cs[, all(round(response, 6) == round(response[1], 6)), by = datetime]
same_predictions[, .N, by = V1]

# Long only
long = back_cs[, data.table::first(.SD), by = datetime]
long[, .N, by = symbol][order(-N)]
long = long[, .(datetime, truth)]
long = long[, lapply(.SD, function(x) Return.cumulative(x)), by = .(datetime = as.Date(datetime))]
long = as.xts.data.table(long)
long = long[long$truth < 0.5]
SharpeRatio.annualized(long, scale = 252)
charts.PerformanceSummary(long)

# Short only
short = back_cs[, data.table::last(.SD), by = datetime] |>
  na.omit() |>
  _[, .(datetime, truth = -truth)] |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(datetime = as.Date(datetime))] |>
  as.xts.data.table()
short = short[short$truth > -0.5]
charts.PerformanceSummary(short)
SharpeRatio.annualized(short, scale = 252)

# Long short
long_short = merge(long, short)
long_short$ret = (long_short$truth + long_short$truth.1) / 2
long_short = long_short[long_short$ret < 0.5]
SharpeRatio.annualized(long_short)
charts.PerformanceSummary(long_short)


half_spread = one_way_cost / 2
back[, weight := NA_integer_]
back[response > half_spread, weight := 1L]
back[response < -half_spread, weight := -1L]
back[is.na(weight)]
setorder(back, symbol, learner_id, datetime)
back[, weight := nafill(weight, type = "locf"), by = .(symbol, learner_id)]
back[, ret_truth := truth * weight]
back[, ret_truth_lead := shift(truth, 1L, type = "lead") * weight]

back[learner_id == "ensamble_median", .(symbol, datetime, truth, ret_truth, ret_truth_net)]
port = back[learner_id == "ensamble_meduab", .(symbol, datetime, truth, ret_truth, ret_truth_net, learner_id)] |>
  dcast(symbol + datetime + truth ~ learner_id, value.var = "ret_truth") |>
  _[, .SD, .SDcols = c("symbol", "datetime", "truth", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(symbol, datetime = as.Date(datetime))]



# ADD TO QC -----------------------------------------------------
# Add data to Quant connect
qc_data = predictions_dt[, .(datetime, learner = learner_id, resp = response)]
qc_data[, jump := 0]
qc_data[grepl("jump", learner), jump := 1]
qc_data[, learner := gsub("filter_jumps\\.", "", learner)]
qc_data[, learner := gsub("\\..*", "", learner)]
qc_data[jump == 1, learner := paste0(learner, "_jump")]
qc_data[, jump := NULL]
models_to_keep = qc_data[, .N > 100000, by = learner][V1 == TRUE, learner]
qc_data = qc_data[learner %in% models_to_keep]
setorder(qc_data, learner, datetime)
qc_data = dcast(qc_data, datetime ~ learner, value.var = "resp")
# keep_cols_with_no_na = !matrixStats::colAnyNAs(as.matrix(qc_data))
# keep_cols_with_no_na = names(keep_cols_with_no_na[keep_cols_with_no_na == TRUE])
qc_data = na.omit(qc_data)
# qc_data = qc_data[, .SD, .SDcols = keep_cols_with_no_na]
qc_data[, ensamble_median := matrixStats::rowMedians(as.matrix(.SD)), .SDcols = is.numeric]
qc_data[, ensamble_mean   := matrixStats::rowMeans2(as.matrix(.SD)), .SDcols = is.numeric]
qc_data[, datetime := as.character(datetime)]
endpoint = storage_endpoint(Sys.getenv("ENDPOINT"), key=Sys.getenv("KEY"))
cont = storage_container(endpoint, "qc-backtest")
storage_write_csv(qc_data, container = cont, file = "intraday_zoo_factors.csv")

# Compare to QC
back[, {
  x = .SD[, .(datetime, ret_target)]
  x = x[, .(ret = Return.cumulative(ret_target)), by = .(date = as.Date(datetime))]  
  x
}, by = learner_id] |>
  _[date %between% c(as.Date("2010-01-04"), as.Date("2012-01-01"))] |>
  _[, learner_id := gsub("\\..*", "", learner_id)] |>
  dcast(formula = date ~ learner_id, value.var = "ret") |>
  as.xts.data.table() |>
  charts.PerformanceSummary()