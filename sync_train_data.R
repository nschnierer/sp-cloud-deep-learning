# Download zip from NextCloud and unzip it.
# Public link:
url = "https://*****.your-storageshare.de/s/*****/download?path=%2Ftrain_data_v2_osm&files=T32UNF_20200807T102559.zip"
download.file(url, "./train_data.zip")
unzip("./train_data.zip")
unlink("./train_data.zip")