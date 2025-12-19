library(data.table)
library(arrow)
library(lubridate)
library(PerformanceAnalytics)
library(roll)
library(highfrequency)
library(ggplot2)


# CONFIG ---------------------------------------------------
# Create data directory if it doesnt exists
PATH_SAVE = file.path(".", "data")

# Parameters
LAST_TIMESTAMP = "15:55:00"


# CLEAN FACTORS --------------------------------------------
# Import FF factors
factors_ff = fread("data/ff6_1min_returns.csv")

# Import 1m factor returns
files = list.files(file.path(PATH_SAVE, "factor_returns"), full.names = TRUE)
factors = lapply(files, read_parquet)
factors = rbindlist(factors)

# Handle timezones
factors[, datetime := force_tz(datetime, tz = Sys.timezone())]
factors[, datetime := with_tz(datetime, tz = "UTC")]
factors[, datetime := force_tz(datetime, tz = "America/New_York")]

# Craete time column
factors[, time := as.ITime(datetime)]

# Create target variable - overnight return
overnight_returns = factors[time == as.ITime("09:30:00"), .(date = as.Date(datetime), datetime, target = ff__mkt)]

# Model specification following HAR logic
predictors = setdiff(colnames(factors), c("datetime", "time"))
cols_5min = paste0(predictors, "_5min")
factors[, (cols_5min) := lapply(.SD, function(x) roll_prod(1 + x, width = 5) - 1), .SDcols = predictors]
cols_10min = paste0(predictors, "_10min")
factors[, (cols_10min) := lapply(.SD, function(x) roll_prod(1 + x, width = 10) - 1), .SDcols = predictors]
cols_15min = paste0(predictors, "_15min")
factors[, (cols_15min) := lapply(.SD, function(x) roll_prod(1 + x, width = 15) - 1), .SDcols = predictors]
cols_30min = paste0(predictors, "_30min")
factors[, (cols_30min) := lapply(.SD, function(x) roll_prod(1 + x, width = 30) - 1), .SDcols = predictors]
cols_hour = paste0(predictors, "_hour")
factors[, (cols_hour) := lapply(.SD, function(x) roll_prod(1 + x, width = 60) - 1), .SDcols = predictors]
cols_2hour = paste0(predictors, "_2hour")
factors[, (cols_2hour) := lapply(.SD, function(x) roll_prod(1 + x, width = 120) - 1), .SDcols = predictors]
cols_4hour = paste0(predictors, "_4hour")
factors[, (cols_4hour) := lapply(.SD, function(x) roll_prod(1 + x, width = 240) - 1), .SDcols = predictors]
cols_day = paste0(predictors, "_day")
factors[, (cols_day) := lapply(.SD, function(x) roll_prod(1 + x, width = 390) - 1), .SDcols = predictors]

# Merge predictors and target
predictors = c(predictors, cols_5min, cols_15min, cols_30min, cols_hour, cols_2hour, cols_4hour, cols_day)
factors_last_timestamp = factors[time == as.ITime(LAST_TIMESTAMP)]
overnight_returns[, datetime_overnight := datetime]
dt = overnight_returns[factors_last_timestamp, on = "datetime", roll = -Inf]

# Remove NA in target
dt = na.omit(dt, cols = "target")

# Inspect dt
dim(dt)
summary(dt[, target])
hist(dt[, target])

# Save final table
fwrite(dt, "data/factor_returns_overnight.csv")
