library(googledrive)
library(data.table)
library(arrow)
library(lubridate)
library(PerformanceAnalytics)
library(roll)
library(highfrequency)


# CONFIG ---------------------------------------------------
# Create data directory if it doesnt exists
PATH_SAVE = file.path(".", "data")


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

# Craete tiem column
factors[, time := as.ITime(datetime)]

# Checks
dim(factors)
colnames(factors)
na.omit(factors[, .(datetime, ff__mkt)][factors_ff[, .(datetime, ff__mkt)], on = "datetime"])
factors[, .N, by = time]

# Remove first return (overnight) Upsample to 15 min
factors_intra = factors[time != as.ITime("09:30:00")]
factors_intra[, interval_15min := ceiling_date(datetime, unit = "15 minutes")]
factors_intra[, .(datetime, interval_15min)]
factor_cols = setdiff(names(factors_intra), c("datetime", "interval_15min", "time"))
factors_intra = factors_intra[
  , lapply(.SD, \(x) prod(1 + x, na.rm = TRUE) - 1),
  by = .(datetime = interval_15min),
  .SDcols = factor_cols
]

# Optional, if log returns
# factors_15 <- factors[
#   , lapply(.SD, \(x) sum(x, na.rm = TRUE)),
#   by = datetime_15,
#   .SDcols = factor_cols
# ]

# Descriptive statistics
if (interactive()) {
  factors_daily = factors_intra[, .(
    mkt_ret = prod(1 + ff__mkt) - 1,
    mkt_rv = sum(ff__mkt^2, na.rm = TRUE)
  ), by = as.Date(datetime)]
  cols_ff6 = factors_intra[, colnames(.SD), .SDcols = patterns("ff__")]
  factors_daily_ff6 = factors_intra[, .(
    ff6_ret = sum(sapply(.SD, function(x) prod(1 + x) - 1)) / 7
  ), by = as.Date(datetime), .SDcols = cols_ff6]
  
  # Cant replicate
  # factors_daily[, sqrt(mkt_rv)]
  # factors_daily[, mean(sqrt(mkt_rv))]
  # factors_daily[, mean(sqrt(mkt_rv)) * sqrt(252)]
  # factors_daily[, mean(sqrt(mkt_rv)* sqrt(252))]
  # factors_daily[, sqrt(mean(mkt_rv)*sqrt(252))]
  # RV = rCov(as.xts.data.table(factors_intra[, .(DT = datetime, ff__mkt)]), periods = 1, alignBy = "days")
  # RV[, sqrt(RV)*sqrt(252)]
  # RV[, mean(sqrt(RV)*sqrt(252))]
  # BPV = rBPCov(factors_intra[, .(DT = datetime, ff__mkt)], periods = 1, alignBy = "days") # Daily Bipower Variation
  # BPV[, sqrt(BPV)*sqrt(252)]
  # BPV[, mean(sqrt(BPV)*sqrt(252))]
  # RV = rCov(factors[, .(DT = datetime, ff__mkt)], periods = 1, alignBy = "days")
  # RV[, sqrt(RV)*sqrt(252)]
  # RV[, mean(sqrt(RV)*sqrt(252))]
  # RV = rRVar(factors_intra[, .(DT = datetime, ff__mkt)], periods = 1, alignBy = "days")
  # RV[, sqrt(RVar)*sqrt(252)]
  # RV[, mean(sqrt(RVar)*sqrt(252))]
  descriptive_from_daily = factors_daily[, .(
    makret_cagr = Return.annualized(mkt_ret, scale = 252),
    market_sd   = StdDev.annualized(mkt_ret , scale = 252),
    market_rv   = sqrt(252)*mean(sqrt(mkt_rv), na.rm = TRUE)
  )]
  descriptive_from_daily_ff6 = factors_daily_ff6[, .(
    ff_cagr     = Return.annualized(ff6_ret, scale = 252)
  )]
  descriptive_from_intra = factors_intra[, .(
    market_autocorr = acf(ff__mkt, lag.max = 1, plot = FALSE)$acf[2]
  )]
  rbind(melt(descriptive_from_daily), melt(descriptive_from_daily_ff6), melt(descriptive_from_intra))
}

# Model specification
predictors = setdiff(colnames(factors_intra), "datetime")
cols_hour = paste0(predictors, "_hour")
factors_intra[, (cols_hour) := lapply(.SD, function(x) roll_prod(1 + x, width = 4) - 1), 
  .SDcols = predictors]
cols_day = paste0(predictors, "_day")
factors_intra[, (cols_day) := lapply(.SD, function(x) roll_prod(1 + x, width = 26) - 1), 
  .SDcols = predictors]

# Create target variable
factors_intra[, target := shift(ff__mkt, 1L, type = "lead")]


# JUMP DETECTION PROCEDURE ---------------------------------
# Utils - EWMA function
ewma = function(x, lambda) {
  ewma = vector(mode = "double", length = length(x))   # initialize output vector
  ewma[1] = x[1]                                        # set first value
  for (i in 2:length(x)) {
    ewma[i] = (1 - lambda) * x[i] + lambda * ewma[i - 1]  # EWMA recursion
  }
  ewma
}

# Parameters
alpha = 3.5      # Threshold parameter (following Bollerslev & Todorov, 2011)
omega = 0.49     # Tuning parameter
n_intraday = factors_intra[, .N, by = as.Date(datetime)][, median(N)]  # Number of observations per day
delta_n = 1 / n_intraday
 
# Caclualte EWMA of exponentially weighted moving average (EWMA) of therolling product of adjacent absolute returns
factors_intra[, sigma := abs(shift(ff__mkt, 1) * shift(ff__mkt, 2))]
factors_intra = na.omit(factors_intra, cols = "sigma")
lambda = exp(log(0.5) / n_intraday)
factors_intra[, sigma := ewma(sigma, lambda)]

# Estimate tau
tod = factors_intra[, .(datetime, time = as.ITime(datetime), ff__mkt)]
setorder(tod, datetime)
tod[, r2 := ff__mkt^2]
tod[, avg_sq_ret_i := roll::roll_mean(r2, width = length(r2), min_obs = 60), by = time]
tod[, avg_sq_ret_day := roll::roll_mean(r2, width = length(r2), min_obs = 60*n_intraday)]
tod[, tau_i := avg_sq_ret_i / avg_sq_ret_day]
tod[, plot(tau_i)]

# Remove columns if exists
factors_intra[, c("tau_i", "jump_threshold", "jump") := NULL]

# Merge factors_intra and tod
factors_intra = tod[, .(datetime, tau_i)][factors_intra, on = "datetime"]

# Estimate final sigma
factors_intra[, sigma := (pi / 2) * n_intraday * sigma * tau_i]
factors_intra[, sigma := sqrt(sigma)]

# Remove missing values
factors_intra = na.omit(factors_intra, cols = "sigma")

# Estimate jump
factors_intra[, jump_threshold := alpha * sigma * (delta_n^omega)]

# Classify jump
factors_intra[, jump := as.integer(abs(ff__mkt) > jump_threshold)]

# Checks
factors_intra[jump == 1, .(datetime, sigma, ff__mkt)]
factors_intra[, .(datetime, sigma, ff__mkt, jump_threshold)]
factors_intra[, sum(jump == 1) / nrow(factors_intra)]

nrow(factors_intra[is.na(sigma)])

x =colnames(factors_intra)
head(x)


