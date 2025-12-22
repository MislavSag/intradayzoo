library(arrow)
library(dplyr)
library(lubridate)
library(ggplot2)
library(patchwork)
library(data.table)


# Globals
PATH         = "F:/data/equity/us/minute-databento/cleaned"
PATH_DATASET = "D:/strategies/intradayzoo"

# Choose symbols
symbols = gsub("symbol=", "", list.files(PATH))
symbols = c(sample(symbols, 20), "SPY")

# Import cleaned daily data
prices = open_dataset(PATH, format = "parquet") |>
  filter(symbol %in% symbols) |>
  collect()
setDT(prices)
prices[, ts_event := with_tz(ts_event, tz = "America/New_York")]

# Filter trading minutes
prices = prices[as.ITime(ts_event) %between% c(as.ITime("09:30:00"), as.ITime("16:00:00"))]

# Upsample to 5 minute frequency
prices = prices[, .(
  open = data.table::first(open),
  high = max(high),
  low = min(low),
  close = data.table::last(close),
  close_raw = data.table::last(close_raw),
  volume = sum(volume)
), by = .(symbol, date = ceiling_date(ts_event, "5 mins"))]

# Compare intraday and all returns
prices[, return := close / shift(close) - 1, by = symbol]
g1 = prices[symbol == "SPY", .(date, return)] |>
  na.omit() |>
  _[, .(date, return, cum_return = cumprod(1 + return) - 1)] |>
  ggplot(aes(x = date, y = cum_return)) +
  geom_line()
g2 = prices[symbol == "SPY" & as.ITime(date) > as.ITime("09:30:00")] |>
  _[, .(date, return)] |>
  na.omit() |>
  _[, .(date, return, cum_return = cumprod(1 + return) - 1)] |>
  ggplot(aes(x = date, y = cum_return)) +
  geom_line()
g1 / g2

# Save every symbol separately
prices_dir = file.path(PATH_DATASET, "prices")
if (!dir.exists(prices_dir)) {
  dir.create(prices_dir)
}
for (s in prices[, unique(symbol)]) {
  prices_ = prices[symbol == s]
  if (nrow(prices_) == 0) next
  file_name = file.path(prices_dir, paste0(s, ".csv"))
  fwrite(prices_, file_name)
}

# Create sh file for predictors
cont = sprintf(
  "#!/bin/bash

#PBS -N pead_predictions
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -J 1-%d
#PBS -o logs
#PBS -j oe

cd ${PBS_O_WORKDIR}

apptainer run image_predictors.sif predictors_padobran.R",
  length(list.files(dataset_dir)))
writeLines(cont, "predictors_padobran.sh")

# Add to padobran
# scp -r /home/sn/data/strategies/pead/prices padobran:/home/jmaric/peadml/prices
