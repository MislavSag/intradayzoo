library(data.table)
library(mlr3batchmark)
library(batchtools)
library(httr)
library(arrow)
library(lubridate)
library(PerformanceAnalytics)
library(AzureStor)
library(finutils)
library(tidyfinance)


# Import registry
reg = loadRegistry("./experiments_overnight", "./experiments_overnight", writeable = TRUE)
getStatus(reg = reg)
job_table = getJobTable(reg = reg)
job_table = unwrap(job_table)
job_table = job_table[,
  .(job.id, learner_id, task_id, resampling_id, repl)
]
findErrors(reg = reg)

# Risk free rate
rf = as.data.table(download_data("factors_ff_5_2x3_daily", "1962-01-01", "2025-10-01"))
rf = as.xts.data.table(rf[, .(date, risk_free)])

# Import results
ids_ = findDone(reg = reg)
bmr = reduceResultsBatchmark(ids_, store_backends = TRUE, reg = reg)

# Aggregate results
bmr$aggregate() |>
  _[order(regr.mse)]

# Detailed results with all measures
bmr_scores = bmr$score(measures = list(msr("regr.mse"), msr("regr.mae"), msr("regr.rmse")))
print(head(bmr_scores))

# Extract predictions
bmr_with_preds = bmr$score(predictions = TRUE)

# Get SPY overnight returns
spy = qc_daily_parquet("C:/Users/Mislav/qc_snp/data/all_stocks_daily", symbols = "spy")
spy = spy[, .(date, target = open / shift(close, 1L) - 1)]
spy[, date := force_tz(date, tzone = "America/New_York")]
spy = na.omit(spy)
spy[, date_spy := date]

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
  datetime_data = task$backend$data(rows = row_ids, cols = c("datetime"))
  datetime_data[, datetime := with_tz(datetime_data[, datetime], tzone = "America/New_York")]
  
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

# Merge predictions_dt and spy
back = spy[predictions_dt, on = c("date" = "datetime"), roll = -Inf]
setnames(back, "date", "datetime")
back = na.omit(back)
back[, learner_id := gsub("\\..*", "", learner_id)]
cols = back[, unique(learner_id)]
back = dcast(back, datetime + row_ids + iteration + target + truth ~ learner_id, value.var = "response") |>
  _[, let(
    ensamble_mean   = matrixStats::rowMeans2(as.matrix(.SD), na.rm = TRUE),
    ensamble_median   = matrixStats::rowMedians(as.matrix(.SD), na.rm = TRUE),
    ensamble_max   = matrixStats::rowMaxs(as.matrix(.SD), na.rm = TRUE),
    ensamble_min   = matrixStats::rowMins(as.matrix(.SD), na.rm = TRUE)
  ), by = datetime, .SDcols = cols] |>
  melt(
    id.vars = c("datetime", "row_ids", "iteration", "target", "truth"), 
    variable.name = "learner_id",
    value.name = "response",
    variable.factor = FALSE
  )

# Approximat trading costs - Conservative estimate (market orders, small size):
round_trip_cost = 0.00002 + 0.00002 + 0.00003  # spread both sides + fees = 0.00007 (0.007%)
one_way_cost = 0.00002 + 0.000015 

# S-sign trading rule
back[, .(mean(target), median(target), mean(truth), median(truth))]
half_spread = one_way_cost / 2
back[, weight := NA_integer_]
back[response > half_spread, weight := 1L]
back[response < -half_spread, weight := -1L]
back[is.na(weight)]
back[, weight := nafill(weight, type = "locf"), by = learner_id]
back[, ret_truth  := truth * weight]
back[, ret_target := target * weight]

# remove na weight
back = na.omit(back, cols = "weight")

# Calculate actual trading costs per period
back[, weight_change := abs(c(NA, diff(ifelse(weight == -1, 0, weight)))), by = learner_id]
back[, trading_cost := weight_change * one_way_cost]
back[is.na(trading_cost), trading_cost := 0]

# Net returns after costs
back[, ret_target_net := ret_target - trading_cost]

# Summary: total costs per learner
back[, .(
  total_trades = sum(weight_change, na.rm = TRUE),
  total_cost_bps = sum(trading_cost, na.rm = TRUE) * 10000,
  avg_cost_per_period_bps = mean(trading_cost, na.rm = TRUE) * 10000
), by = learner_id]

# Predicttions to wide format
back[, .N, by = learner_id]
models_to_keep = back[, .N > 4000, by = learner_id][V1 == TRUE, learner_id]
port_mkt  = back[, .(datetime, ret_truth, ret_target, ret_target_net, learner_id)] |>
  dcast(datetime ~ learner_id, value.var = "ret_truth") |>
  _[, .SD, .SDcols = c("datetime", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(date = as.Date(datetime))] |>
  as.xts.data.table()
port_spy  = back[, .(datetime, ret_truth, ret_target, ret_target_net, learner_id)] |>
  dcast(datetime ~ learner_id, value.var = "ret_target") |>
  _[, .SD, .SDcols = c("datetime", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(date = as.Date(datetime))] |>
  as.xts.data.table()
port_spy_net  = back[, .(datetime, ret_truth, ret_target, ret_target_net, learner_id)] |>
  dcast(datetime ~ learner_id, value.var = "ret_target_net") |>
  _[, .SD, .SDcols = c("datetime", models_to_keep)] |>
  na.omit() |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(date = as.Date(datetime))] |>
  as.xts.data.table()

# Performance
rf_ = merge.xts(port_spy, rf, join = "left")[, "risk_free"]
melt(as.data.table(SharpeRatio.annualized(port_mkt, scale = 252, Rf = rf_)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252, Rf = rf_)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(port_spy_net, scale = 252, Rf = rf_)))[order(-value)]

# Biggest decrease gorss / net
x = melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252)))
y = melt(as.data.table(SharpeRatio.annualized(port_spy_net, scale = 252)))  # Net of actual costs
cbind(x[, 1], x[, 2] - y[, 2])[order(value)]

# Capital curves
charts.PerformanceSummary(port_mkt)
charts.PerformanceSummary(port_spy)
charts.PerformanceSummary(port_spy_net)  # Net of actual costs
charts.PerformanceSummary(port_spy["2007/2008"])

# Add data to Quant connect
qc_data = predictions_dt[, .(datetime, learner = learner_id, resp = response)]
qc_data[, learner := gsub("\\..*", "", learner)]
models_to_keep = qc_data[, .N > 4000, by = learner][V1 == TRUE, learner]
qc_data = qc_data[learner %in% models_to_keep]
setorder(qc_data, learner, datetime)
qc_data = dcast(qc_data, datetime ~ learner, value.var = "resp")
qc_data = na.omit(qc_data)
qc_data[, ensamble_median := matrixStats::rowMedians(as.matrix(.SD)), .SDcols = is.numeric]
qc_data[, ensamble_mean   := matrixStats::rowMeans2(as.matrix(.SD)), .SDcols = is.numeric]
qc_data[, datetime := as.character(datetime)]
endpoint = storage_endpoint(Sys.getenv("ENDPOINT"), key=Sys.getenv("KEY"))
cont = storage_container(endpoint, "qc-backtest")
storage_write_csv(qc_data, container = cont, file = "overnight_zoo_factors.csv")

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

# Checks
plot(as.xts.data.table(predictions_dt[learner_id == "earth.tuned", .(datetime, response)][1:1000]))
ask = 101
bid = 100
spread = ask - bid
midprice = (ask + bid) / 2
spread / midprice


# ARCHIVE ----------------------------------------------------------------------
# endpoint = "https://financialmodelingprep.com/stable/historical-chart/1min"
# dates = seq.Date(as.Date("2004-01-01"), Sys.Date(), by = 1)
# spy_l = lapply(dates, function(d) {
#   Sys.sleep(0.1)
#   p = GET(
#     url = endpoint,
#     query = list(
#       symbol = "SPY",
#       from=d,
#       to=d,
#       apikey="6c390b6279b8a67cc7f63cbecbd69430"
#     )
#   )
#   res = content(p)
#   if (length(res) == 0) return(NULL)
#   dt_ = lapply(res, as.data.table)
#   dt_ = rbindlist(dt_)
#   dt_
# })
# spy = rbindlist(spy_l)

