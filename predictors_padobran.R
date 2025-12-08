options(progress_enabled = FALSE)

library(data.table)
library(finfeatures)
library(reticulate)


# python environment
# reticulate::use_virtualenv("/home/sn/projects_py/pyquant", required = TRUE)
# theftms::init_theft("/opt/venv")
reticulate::use_virtualenv("/opt/venv")
tsfel = reticulate::import("tsfel")
tsfresh = reticulate::import("tsfresh", convert = FALSE)
warnigns = reticulate::import("warnings", convert = FALSE)
warnigns$filterwarnings('ignore')

# paths
PATH_PREDICTORS = file.path("./predictors_padobran")

# Create directory if it doesnt exists
if (!dir.exists(PATH_PREDICTORS)) {
  dir.create(PATH_PREDICTORS)
}

# Get index
i = as.integer(Sys.getenv('PBS_ARRAY_INDEX'))
# i = 1

# Get symbol
# symbols = gsub("\\.csv", "", list.files("/home/sn/data/strategies/pead/dataset"))
symbols = gsub("\\.csv", "", list.files("dataset"))
symbol_i = symbols[i]

# Get data
# dataset = fread(file.path("/home/sn/data/strategies/pead/dataset", paste0(symbol_i, ".csv")))
# ohlcv = fread(file.path("/home/sn/data/strategies/pead/prices", paste0(symbol_i, ".csv")))
dataset = fread(file.path("dataset", paste0(symbol_i, ".csv")))
ohlcv = fread(file.path("prices", paste0(symbol_i, ".csv")))

# Create Ohlcv object
ohlcv = Ohlcv$new(ohlcv[, .(symbol, date, open, high, low, close, volume)], date_col = "date")

# Lag parameter
# ako je red u events amc. label je open_t+1 / close_t; lag je 1L
# ako je red u events bmo. label je open_t / close_t-1; lag je 2L
# radi jednostavnosti lag_ = 2L
lag_ = -1L

# Window
# Beaware of data size
workers = 1L

# Default windows. Set widnows you use the most in rolling predictors
windows = c(66, 252) # day and 2H;  cca 10 days

# Define at parameter
at_ = merge(ohlcv$X[, .(symbol, date)],
            dataset[, .(symbol, date, index = TRUE)],
            by = c("symbol", "date"), all.x = TRUE, all.y = FALSE)
at_ = at_[, which(index == TRUE)]

# Help function to save rolling predictors output
create_path = function(name) {
  file.path(PATH_PREDICTORS, paste0(name, "-", symbol_i, ".csv"))
}

# Exuber
path_ =   create_path("exuber")
windows_ = c(windows, 504)
if (max(at_) > min(windows)) {
  exuber_init = RollingExuber$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    exuber_lag = c(1L)
  )
  exuber = exuber_init$get_rolling_features(ohlcv, log_prices = TRUE)
  fwrite(exuber, path_)
}

# Backcusum
path_ = create_path("backcusum")
if (max(at_) > min(windows)) {
  backcusum_init = RollingBackcusum$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    alternative = c("greater", "two.sided"),
    return_power = c(1, 2))
  backcusum = backcusum_init$get_rolling_features(ohlcv)
  fwrite(backcusum, path_) 
}

# Forecasts
path_ = create_path("forecasts")
windows_ = c(252, 252 * 2)
if (max(at_) > min(windows)) {
  forecasts_init = RollingForecats$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    forecast_type = c("autoarima", "nnetar", "ets"),
    h = 22)
  forecasts = suppressMessages(forecasts_init$get_rolling_features(ohlcv))
  fwrite(forecasts, path_)
}

# Theft r
path_ = create_path("theftr")
windows_ = c(5, 22, windows)
if (max(at_) > min(windows)) {
  theft_init = RollingTheft$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    features_set = c("catch22", "feasts"))
  theft_r = theft_init$get_rolling_features(ohlcv)
  fwrite(theft_r, path_)
}

# Theft r with returns
path_ = create_path("theftrr")
windows_ = c(5, 22, windows)
if (max(at_) > min(windows)) {
  theft_init = RollingTheft$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    features_set = c("catch22", "feasts"))
  theft_r = theft_init$get_rolling_features(ohlcv, price_col = "returns")
  fwrite(theft_r, path_)
}

# Theft py with returns
path_ = create_path("theftpy")
windows_ = c(22, windows)
if (max(at_) > min(windows)) {
  theft_init = RollingTheft$new(
    windows = windows_,
    workers = 1L,
    at = at_,
    lag = lag_,
    features_set = c("tsfel", "tsfresh"))
  theft_py = suppressMessages(theft_init$get_rolling_features(ohlcv, price_col = "returns"))
  fwrite(theft_py, path_)
}

# Theft py
path_ = create_path("theftpyr")
windows_ = c(22, windows)
if (max(at_) > min(windows)) {
  theft_init = RollingTheft$new(
    windows = windows_,
    workers = 1L,
    at = at_,
    lag = lag_,
    features_set = c("tsfel", "tsfresh"))
  theft_py = suppressMessages(theft_init$get_rolling_features(ohlcv))
  fwrite(theft_py, path_)
}

# Tsfeatures
path_ = create_path("tsfeatures")
if (max(at_) > min(windows)) {
  tsfeatures_init = RollingTsfeatures$new(
    windows = windows,
    workers = workers,
    at = at_,
    lag = lag_,
    scale = TRUE)
  tsfeatures = suppressMessages(tsfeatures_init$get_rolling_features(ohlcv))
  fwrite(tsfeatures, path_)
}

# WaveletArima
path_ = create_path("waveletarima")
if (max(at_) > min(windows)) {
  waveletarima_init = RollingWaveletArima$new(
    windows = windows,
    workers = workers,
    at = at_,
    lag = lag_,
    filter = "haar")
  waveletarima = suppressMessages(waveletarima_init$get_rolling_features(ohlcv))
  fwrite(waveletarima, path_)
}

# FracDiff
path_ = create_path("fracdiff")
if (max(at_) > min(windows)) {
  fracdiff_init = RollingFracdiff$new(
    windows = windows,
    workers = workers,
    at = at_,
    lag = lag_,
    nar = c(1), 
    nma = c(1),
    bandw_exp = c(0.1, 0.5, 0.9))
  fracdiff = suppressMessages(fracdiff_init$get_rolling_features(ohlcv))
  fwrite(fracdiff, path_)
}

# VSE
path_ = create_path("vse")
windows_ = c(22, 44, 150, windows, 504)
if (max(at_) > min(windows)) {
  vse_init = RollingVse$new(
    windows = windows_,
    workers = workers,
    at = at_,
    lag = lag_,
    m = c(0.4, 0.5))
  vse = vse_init$get_rolling_features(ohlcv)
  fwrite(vse, path_)
}
