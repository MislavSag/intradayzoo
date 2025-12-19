library(data.table)
library(mlr3batchmark)
library(batchtools)
library(httr)
library(arrow)
library(lubridate)
library(PerformanceAnalytics)
library(AzureStor)


# Import registry
reg = loadRegistry("./experiments", "./experiments", writeable = TRUE)
getStatus(reg = reg)
findDone(reg = reg)
job_table = getJobTable(reg = reg)
job_table = unwrap(job_table)
job_table = job_table[,
  .(job.id, learner_id, task_id, resampling_id, repl)
]
findErrors(reg = reg)

# Import results
ids_ = findDone(reg = reg)
remove_ids = 17*c(1:6)
ids_ = ids_[job.id %notin% remove_ids]
bmr = reduceResultsBatchmark(ids_, store_backends = TRUE, reg = reg)

# Aggregate results
bmr$aggregate() |>
  _[order(regr.mse)]

# Detailed results with all measures
bmr_scores = bmr$score(measures = list(msr("regr.mse"), msr("regr.mae"), msr("regr.rmse")))
print(head(bmr_scores))

# Extract predictions
bmr_with_preds = bmr$score(predictions = TRUE)

# Get SPY 15 min data
spy = read_parquet(file.path("data", "spy_minute.parquet"))
setDT(spy)
spy[, date := force_tz(date, tzone = Sys.timezone())]
spy[, date := with_tz(date, tzone = "UTC")]
spy[, date := force_tz(date, tzone = "America/New_York")]
attr(spy$date, "tz")

# Adjust open and keep columns we need
spy[, open_adj := close_adj / close * open]
spy[, c("open", "high", "low", "volume") := NULL]
setnames(spy, c("close_raw", "close", "date", "open"))
setcolorder(spy, c("date", "open", "close", "close_raw"))

# Remove first return (overnight) Upsample to 15 min
spy = spy[as.ITime(date) != as.ITime("09:30:00")]
spy[, interval_15min := ceiling_date(date, unit = "15 minutes")]
spy[, .(date, interval_15min)]
spy = spy[, .(
  open = data.table::first(open),
  close = data.table::last(close),
  close_raw = data.table::last(close_raw)
), by = .(datetime = interval_15min)]

# Calculate returns
spy[, target := close / open - 1]
spy[, target := shift(target, 1L, type= "lead")]

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
attr(spy$datetime, "tz")
attr(predictions_dt$datetime, "tz")
back = spy[, .(datetime, target)][predictions_dt, on = "datetime"]
back = na.omit(back)
back[, jump := 0]
back[grepl("jump", learner_id), jump := 1]
back[, learner_id := gsub("filter_jumps\\.", "", learner_id)]
back[, learner_id := gsub("\\..*", "", learner_id)]
back[jump == 1, learner_id := paste0(learner_id, "_jump")]
back[, jump := NULL]
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
models_to_keep = back[, .N > 100000, by = learner_id][V1 == TRUE, learner_id]
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
melt(as.data.table(SharpeRatio.annualized(port_mkt, scale = 252)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252)))[order(-value)]
melt(as.data.table(SharpeRatio.annualized(port_spy_net, scale = 252)))[order(-value)]  # Net of actual costs

# Biggest decrease gorss / net
x = melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252)))
y = melt(as.data.table(SharpeRatio.annualized(port_spy_net, scale = 252)))  # Net of actual costs
cbind(x[, 1], x[, 2] - y[, 2])[order(value)]

# Capital curves
charts.PerformanceSummary(port_mkt)
charts.PerformanceSummary(port_spy)
charts.PerformanceSummary(port_spy_net)  # Net of actual costs
charts.PerformanceSummary(port_spy["2007/2008"])

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

