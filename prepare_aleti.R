library(googledrive)


# CONFIG ---------------------------------------------------
# Authenticate google drive
drive_auth(scopes = "https://www.googleapis.com/auth/drive")

# Create data directory if it doesnt exists
PATH_SAVE = file.path(".", "data")
if (!dir.exists(PATH_SAVE)) dir.create(PATH_SAVE)
if (!dir.exists(file.path(PATH_SAVE, "factor_returns"))) dir.create(file.path(PATH_SAVE, "factor_returns"))


# ALETI DATA -----------------------------------------------
# Get Fama-French factors
folder_id = as_id("1SvCqDR_pTbteo8IRvEnbyShwI2JptMG2")
files = drive_ls(folder_id)
file_name = file.path(PATH_SAVE, files$name)
drive_download(files$name, path = file_name, overwrite = TRUE)

# Get all factors by date
folder_id_factor_returns_1m = as_id("1nlwZF18V8x8pTcmnH9S0nZCmdruFLniE")
files_factor_returns = drive_ls(folder_id_factor_returns_1m)
for (i in 1:nrow(files_factor_returns)) {
  print(i)
  file_name = file.path(PATH_SAVE, "factor_returns", files_factor_returns$name[i])
  if (file.exists(file_name)) next
  id = files_factor_returns$id[i]
  drive_download(file = id, path = file_name, overwrite = TRUE)
}

