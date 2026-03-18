library(data.table)

# Import fi
experiment_path = "experiments_fi"
fi = fread(file.path(experiment_path, "fi.csv"))

fi[, .N, by = V1] |> 
  _[order(-N)] |>
  head(n = 10)
