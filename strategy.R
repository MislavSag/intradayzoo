library(data.table)
library(mlr3batchmark)
library(batchtools)
library(httr)


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
endpoint = "https://financialmodelingprep.com/stable/historical-chart/1min"
dates = seq.Date(as.Date("2004-01-01"), Sys.Date(), by = 1)
spy_l = lapply(dates, function(d) {
  Sys.sleep(0.1)
  p = GET(
    url = endpoint,
    query = list(
      symbol = "SPY",
      from=d,
      to=d,
      apikey="6c390b6279b8a67cc7f63cbecbd69430"
    )
  )
  res = content(p)
  if (length(res) == 0) return(NULL)
  dt_ = lapply(res, as.data.table)
  dt_ = rbindlist(dt_)
  dt_
})
spy = rbindlist(spy_l)

