# Data Dictionary

The acquired official CSV has 19,735 rows, 29 columns and no missing cells. Source definitions describe temperatures in °C, relative humidity in %, pressure in mm Hg, windspeed in m/s, visibility in km, and appliance/lighting energy in Wh per 10-minute interval. `date` is timestamp; `Appliances` is the target; `lights` is lighting use.

| Columns | Definition / unit |
|---|---|
| date | Timestamp, nominally 10-minute sampling |
| Appliances | Appliance use, Wh per interval |
| lights | Lighting use, Wh per interval |
| T1–T9 | Indoor zone temperatures, °C |
| RH_1–RH_9 | Indoor zone relative humidity, % |
| T_out | Outdoor temperature, °C |
| Press_mm_hg | Atmospheric pressure, mm Hg |
| RH_out | Outdoor relative humidity, % |
| Windspeed | Wind speed, m/s |
| Visibility | Visibility, km |
| Tdewpoint | Outdoor dew point, °C |
| rv1, rv2 | Source random variables; no physical unit |

Engineered features in the recorded 63-column schema: `appliances_current` is Appliances at t; `appliances_lag_k` is k samples before t; `appliances_roll_mean_w` and `appliances_roll_std_w` summarize the preceding w readings (shifted to exclude t); `*_change_1` fields are current minus previous values for Appliances and selected sensors; calendar fields include hour, weekday, weekend, minute of day, and daily/weekly sine and cosine terms. `target_next` is Appliances at the next observed row, usually t+10 minutes. All lags, rolling summaries and differences use information available at or before the forecast origin. Warm-up rows without complete required history are removed.

Source and attribution: UCI Appliances Energy Prediction, https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction; Candanedo, Feldheim & Deramaix (2017), DOI 10.1016/j.enbuild.2017.01.083.
