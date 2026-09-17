# Data

## Source

**Rossmann Store Sales** — Kaggle competition.
<https://www.kaggle.com/c/rossmann-store-sales/data>

Two files are used, `train.csv` and `store.csv`. Kaggle's competition rules
don't permit redistributing the raw data, so it isn't committed to this
repo — `data/raw/` and `data/processed/` are gitignored except for a
`.gitkeep` placeholder, so the folder structure is preserved but the data
itself isn't tracked.

## How to (re)download it

1. Accept the competition rules on the [Kaggle competition page](https://www.kaggle.com/c/rossmann-store-sales/rules) (required once, even for a closed/completed competition).
2. Either download `train.csv` and `store.csv` manually from the [data tab](https://www.kaggle.com/c/rossmann-store-sales/data) and place them in `data/raw/`, or, with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured:

```bash
kaggle competitions download -c rossmann-store-sales -p data/raw/
cd data/raw && unzip rossmann-store-sales.zip && cd ../..
```

## Raw shape

- **`train.csv`:** 1,017,209 rows × 9 columns — one row per store, per day.
- **`store.csv`:** 1,115 rows × 10 columns — static metadata per store (type, assortment, competition distance, etc.), joined onto `train.csv` via `Store`.
- **Merged:** 1,017,209 rows × 18 columns.
- **Date range:** 2013-01-01 to 2015-07-31 (2.5 years), across 1,115 stores.
- **Closed-store days:** 17.0% of rows have `Open = 0` (`Sales` is always 0 on these days). These are dropped for modeling and for most EDA about sales *levels*, leaving **844,392 rows**.
- **Missing values (in the merged table):** `Promo2SinceWeek` / `PromoInterval` / `Promo2SinceYear` are missing for 508,031 rows (stores that never ran the secondary "Promo2" campaign); `CompetitionOpenSinceMonth` / `CompetitionOpenSinceYear` are missing for 323,348 rows; `CompetitionDistance` is missing for 2,642 rows. These are all legitimate "not applicable" gaps rather than data-quality errors, and are handled accordingly in feature engineering (see `notebooks/02...ipynb`).

## Column reference

### `train.csv`

| Column | Type | Description |
|---|---|---|
| `Store` | Integer | Unique store id (1–1115) |
| `DayOfWeek` | Integer (1–7) | Day of week, Kaggle's own encoding (superseded by `src/features.py`'s 0-indexed `DayOfWeek`) |
| `Date` | Date | Calendar date |
| `Sales` | Integer (target) | Turnover for that store on that day |
| `Customers` | Integer | Number of customers that day (not used as a model feature — it's not known in advance at forecast time) |
| `Open` | Binary | 0 = closed, 1 = open |
| `Promo` | Binary | Whether a store was running a daily promotion that day |
| `StateHoliday` | Categorical | `a` = public holiday, `b` = Easter, `c` = Christmas, `0` = none |
| `SchoolHoliday` | Binary | Whether the store was affected by a public school closure |

### `store.csv`

| Column | Type | Description |
|---|---|---|
| `Store` | Integer | Unique store id |
| `StoreType` | Categorical | Store model, `a`–`d` |
| `Assortment` | Categorical | Assortment level: `a` = basic, `b` = extra, `c` = extended |
| `CompetitionDistance` | Float | Distance (meters) to the nearest competitor store |
| `CompetitionOpenSince[Month/Year]` | Float | When the nearest competitor opened |
| `Promo2` | Binary | Whether the store participates in "Promo2", a recurring secondary promotion |
| `Promo2Since[Week/Year]` | Float | When the store started participating in Promo2 |
| `PromoInterval` | Categorical | Which months Promo2 re-starts in |

## Processed data

`src/data_loader.load_clean_data()` merges the two files, drops closed-store
days, and returns the 844,392-row working table used throughout the project.
`src/features.build_feature_pipeline()` then adds calendar, lag, and rolling
features (see `notebooks/02_feature_engineering_and_baseline_model.ipynb`)
and drops the ~4.0% of rows with insufficient per-store history for the lag
features, leaving 810,942 rows. Nothing is cached to `data/processed/` by
default since re-running the pipeline is fast; the folder exists for anyone
who wants to save an intermediate CSV locally.
