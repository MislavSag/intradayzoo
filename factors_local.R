library(data.table)
library(AzureStor)
library(lubridate)
library(janitor)
library(roll)
library(ggplot2)


# Azure creds
endpoint = storage_endpoint(Sys.getenv("ENDPOINT"), Sys.getenv("KEY"))
cont = storage_container(endpoint, "factors")

# Import util
path = file.path("data", "factors_local")
if (!dir.exists(path)) dir.create(path)
import_factors = function(x) {
  # x = "equal_weight_market_15min.csv"
  file_path = file.path(path, x)
  if (file.exists(file_path)) {
    dt_ = fread(file_path)
  } else {
    dt_  = storage_read_csv(cont, x)
    setDT(dt_)
    if (any(grepl("market_return", colnames(dt_)))) {
        setnames(dt_, "market_return", "ff__mkt")
      }
    fwrite(dt_, file_path)
  }
  dt_[, date := force_tz(date, tz = "America/New_York")]
  dt_
}

# Import all data
AzureStor::list_blobs(cont)
equal_weight_industry_49_15min = import_factors("equal_weight_industry_49_15min.csv")
equal_weight_market_15min      = import_factors("equal_weight_market_15min.csv")
factors_intra = merge(equal_weight_market_15min, equal_weight_industry_49_15min, by = "date")

# Model specification 
predictors = setdiff(colnames(factors_intra), "date")
cols_hour = paste0(predictors, "_hour")
factors_intra[, (cols_hour) := lapply(.SD, function(x) roll_prod(1 + x, width = 4) - 1), 
  .SDcols = predictors]
cols_day = paste0(predictors, "_day")
factors_intra[, (cols_day) := lapply(.SD, function(x) roll_prod(1 + x, width = 26) - 1), 
  .SDcols = predictors]

# Merge all factors together
factors_intra = remove_empty(factors_intra, which = "cols", cutoff = 0.97, quiet = FALSE)

# Remove NA values
factors_intra = na.omit(factors_intra)

# Rename date to datetime
setnames(factors_intra, "date", "datetime")


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
factors_intra[, .(J = sum(jump)), by = .(date = as.Date(datetime))] |>
  _[, .(date, N = frollsum(J, n = 252))] |>
  na.omit() |>
  ggplot(aes(date, N)) +
  geom_line() # very similar to the paper!


# CONT AND JUMP TARGET ----------------------------------------------
# Remove temp columns
colnames(factors_intra)
factors_intra[, c("targetj", "targetc", "ret_total", "ret_total_j", "ret_total_c") := NULL]

# Crate returns, jump returns and cont returns
factors_intra[, ret_total   := ff__mkt]
factors_intra[jump == 1, ret_j := ret_total]
factors_intra[jump == 0, ret_c := ret_total]

# Check
returns = factors_intra[, .(datetime, ret_total, ret_j, ret_c)]
returns[, lapply(.SD, function(x) mean(x, na.rm = TRUE) * 100), .SDcols = -1] |>
  melt() |>
  ggplot(aes(variable, value)) +
  geom_bar(stat = "identity")

# Create target variable for all returns
factors_intra[, target  := shift(ret_total, 1L, type = "lead")]
factors_intra[, targetc := shift(ret_c, 1L, type = "lead")]

# Remove temp columns
colnames(factors_intra)
factors_intra[, c("ret_total", "ret_j", "ret_c", "tau_i", "sigma", "jump_threshold") := NULL]

# Save final table
fwrite(factors_intra, "data/factor_returns_local.csv")
