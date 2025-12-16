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
back = spy[, .(datetime, target, target_lag)][predictions_dt, on = "datetime"]
back = na.omit(back)

# S-sign trading rule
back[, .(mean(target), median(target), mean(truth), median(truth))]
half_spread = 0.00002
back[, weight := NA_integer_]
back[response > half_spread, weight := 1L]
back[response < -half_spread, weight := -1L]
back[is.na(weight)]
back[, weight := nafill(weight, type = "locf"), by = learner_id]
back[, ret_truth  := truth * weight]
back[, ret_target := target * weight]

# Predicttions to wide format
port_mkt  = back[, .(datetime, ret_truth, ret_target, learner_id)] |>
  _[, learner_id := gsub("\\..*", "", learner_id)] |>
  dcast(datetime ~ learner_id, value.var = "ret_truth") |>
  _[, .SD, .SDcols = -"xgboost"] |>
  na.omit() |>
  _[, ensamble := matrixStats::rowMeans2(as.matrix(.SD)), .SDcols = is.numeric] |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(date = as.Date(datetime))] |>
  as.xts.data.table()
port_spy  = back[, .(datetime, ret_truth, ret_target, learner_id)] |>
  _[, learner_id := gsub("\\..*", "", learner_id)] |>
  dcast(datetime ~ learner_id, value.var = "ret_target") |>
  _[, .SD, .SDcols = -"xgboost"] |>
  na.omit() |>
  _[, ensamble := matrixStats::rowMeans2(as.matrix(.SD)), .SDcols = is.numeric] |>
  _[, lapply(.SD, function(x) Return.cumulative(x)), by = .(date = as.Date(datetime))] |>
  as.xts.data.table() 
  
# Performance
melt(as.data.table(SharpeRatio.annualized(port_mkt, scale = 252)))
melt(as.data.table(SharpeRatio.annualized(port_spy, scale = 252)))

# Capital curves
charts.PerformanceSummary(port_mkt)
charts.PerformanceSummary(port_spy)
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
qc_data[, learner := gsub("\\..*|_.*", "", learner)]
setorder(qc_data, learner, datetime)
qc_data = dcast(qc_data, datetime ~ learner, value.var = "resp")
keep_cols_with_no_na = !matrixStats::colAnyNAs(as.matrix(qc_data))
keep_cols_with_no_na = names(keep_cols_with_no_na[keep_cols_with_no_na == TRUE])
qc_data = qc_data[, .SD, .SDcols = keep_cols_with_no_na]
qc_data[, ensamble := matrixStats::rowMeans2(as.matrix(.SD)), .SDcols = is.numeric]
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

