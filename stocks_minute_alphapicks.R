library(arrow)
library(data.table)
library(httr)
library(fasttime)
library(lubridate)
use("dplyr", c("select", "distinct", "collect", "pull", "filter"))

# Config
PATH = "data/ohlcv_1min_events"
if (!dir.exists(PATH)) dir.create(PATH) 

# Minute data function
get_data = function(uri, symbol, start_date = Sys.Date() - 5, end_date = Sys.Date()) { 
  # uri   = PATH
  # symbol = "AAPL"

  # check dir
  if (!dir.exists(uri)) stop("Dir doesnt exists")

  # Try to open existing dataset and filter for this symbol
  if (length(list.files(uri)) > 0) {
    ds = open_dataset(uri)
    
    # Check if symbol column exists and get existing symbols
    if ("symbol" %in% names(ds)) {
      existing_symbols = ds |> 
        select(symbol) |> 
        distinct() |> 
        collect() |> 
        pull(symbol)
      
      if (symbol %in% existing_symbols) {
        message(paste("Loading existing data for", symbol))
        dt_ = ds |> 
          filter(symbol == !!symbol) |> 
          collect() |> 
          as.data.table()
        return(dt_)
      }
    }
  }
  
  # If not exists, fetch the data
  endpoint = "https://financialmodelingprep.com/stable/historical-chart/1min"
  dates = seq.Date(start_date, end_date, by = 1)
  dt_l = lapply(dates, function(d) {
    Sys.sleep(0.06)
    p = GET(
      url = endpoint,
      query = list(
        symbol = symbol,
        from = d,
        to = d,
        apikey = Sys.getenv("FMP")
      )
    )
    res = content(p)
    if (length(res) == 0) return(NULL)
    dt_ = lapply(res, as.data.table)
    dt_ = rbindlist(dt_, fill = TRUE)
    if (!is.null(dt_) && nrow(dt_) > 0) {
      dt_[, symbol := symbol]
    }
    dt_
  })
  dt_ = rbindlist(dt_l, fill = TRUE)
  
  if (nrow(dt_) == 0) {
    message(paste("No data fetched for", symbol))
    return(dt_)
  }
  
  # Save to Arrow dataset with partitioning by symbol
  message(paste("Saving data for", symbol, "to Arrow dataset"))
  dt_ |> 
    arrow::write_dataset(
      path = uri,
      partitioning = "symbol",
      format = "parquet"
    )
  
  return(dt_)
}

# Get Ohlcv data from FMP
picks = fread("data/alpha_picks.csv")
pqp = fread("data/pqp.csv")
symbols = unique(picks[, Ticker], pqp[, Ticker])
for (symbol in symbols) {
  get_data(uri = PATH, symbol = symbol, start_date = as.Date("2003-09-01"), end_date = as.Date("2025-12-20"))
}

# Import data
ohlcv = lapply(symbols, function(x) get_data(PATH, x, start_date = as.Date("2003-09-01"), end_date = as.Date("2025-12-20")))
ohlcv = rbindlist(ohlcv)
ohlcv[, date := fastPOSIXct(date, tz = "GMT")]
ohlcv[, date := force_tz(date, tzone = "America/New_York")]

# Plot daily data to check for errors in prices
ohlcv[, .(close = data.table::last(close)), by = .(symbol, date = as.IDate(date))] |>
  dcast(date ~ symbol, value.var = "close") |>
  as.xts.data.table() |>
  plot()

# Save 
# write_parquet(ohlcv, "data/stocks_minute.parquet")

# Remove overnight and upasample to 15 minute
ohlcv[, time := as.ITime(date)]
ohlcv_intra = ohlcv[time != as.ITime("09:30:00")]
ohlcv_intra[, interval_15min := ceiling_date(date, unit = "15 minutes")]
ohlcv_intra = ohlcv_intra[, .(
  open  = data.table::first(open),
  close = data.table::last(close)
), by = .(symbol, date = interval_15min)]
ohlcv_intra[, target := close / open - 1]
setorder(ohlcv_intra, symbol, date)
# ohlcv_intra[, .(date, target, shift(target, 1L, "lead")), by = symbol]
# ohlcv_intra[, target := shift(target, 1L, "lead"), by = symbol]
ohlcv_intra = ohlcv_intra[, .(symbol, date, target)]

# Save
fwrite(ohlcv_intra, "data/ohlcv_15min.csv")
