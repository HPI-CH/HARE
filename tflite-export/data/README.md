# Data directory
In our thesis we have used two kinds of data sets: 
- `raw`: sensor data is not merged into a single file, but there is one file per sensor as well as one metadata json file containing subject, activity per time interval etc. per recording. This means that there is one directory containing the mentioned files for each recording. This is the format that our app uses for storing recordings.
- `merged`: sensor data as well as the activity per time stamp is merged into a single CSV file, meaning each row contains data from all the used sensors as well as the corresponding activity labelled for the person wearing the sensors at that point in time.  

## TFLite Export data sets (merged)
To execute code in [tflite-export](../../tflite-export/) you need to acquire some data sets. Here we use `merged` data sets for ease of use and performance advantages over `raw` data sets.

1. download the `merged` data sets `opportunity`, `gait` and `nursing` [here](https://nextcloud.hpi.de/s/fSKsgwQ2bx2DRWs). They are archived in `tflite_export_datasets.zip`. 
2. unpack the zip archive to your local repository into [this directory](.) so that the directories `tflite-export/data/nursing`, `tflite-export/data/opportunity` and `tflite-export/data/gait` exist.

## On Device data sets (raw)
To perform on-device inference and training, recordings are needed. These recordings need to be in the format that the app understands, meaning the format that the app itself stores its recordings in, namely the `raw` format. Since we only recorded the nursing data set using our app, we also only have the nursing data set in `raw` format (download `nursing_raw` from [here](https://nextcloud.hpi.de/s/fSKsgwQ2bx2DRWs)).
To import these recordings into the app, copy the extracted recording directories (each recording directory contains one `metadata.json` file as a direct child) into `Pixel 6\Internal shared storage\Android\data\sensors_in_paradise.xsens\files\useCases\[USE_CASE]\recordings\[RECORDING_SUBDIR]`. Replace `USE_CASE` and `RECORING_SUBDIR` with the use case and recording subdir you selected in the app. For the default use case and sub dir the path would be `Pixel 6\Internal shared storage\Android\data\sensors_in_paradise.xsens\files\useCases\default\recordings\default`. 
You can then perform inference and personalization on the app via the `data` screen.

For the Gait data set (which we also performed on-device personalization with during our thesis), we only have the `merged` version. 
Since our app merges the `raw` recordings into `merged` CSV-files before performing inference or training on them, one can also make use of these merged files. However the app UI does not display merged recordings, nor does the app detect those as recordings. Therefor one can write instrumentation tests to use our app code to directly load these merged recordings and perform ML operations on them. This is what we did for the Gait experiments as well. 
