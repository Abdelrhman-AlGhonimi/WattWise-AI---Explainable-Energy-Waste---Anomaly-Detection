# Data

The raw CSV is acquired from the official UCI Appliances Energy Prediction archive by `wattwise.data_loader`. Local raw and processed data are git-ignored. The acquisition records a source URL and download timestamp beside the CSV. Manual recovery: download the official dataset archive at https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction, extract `energydata_complete.csv` to `data/raw/energydata_complete.csv`, and run the training command. Preserve this source file unchanged.

Dataset: 19,735 timestamped records, 29 columns. Appliances is measured in Wh over each 10-minute period. See `docs/DATA_DICTIONARY.md`. UCI attribution: Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017), Data driven prediction models of energy use of appliances in a low-energy house, Energy and Buildings 140, 81–97. DOI 10.1016/j.enbuild.2017.01.083.
