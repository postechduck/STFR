# Data and evaluation

## Datasets and sources

| Dataset | Source | Preprocessing | Users | Items |
| --- | --- | --- | --- | --- |
| Amazon-VG | Amazon Review Data (2018), Video Games, 5-core review file `Video_Games_5.json.gz` (497,577 reviews) | inherited 5-core filter, dense ids, first interaction per (user, item) | 55,144 | 17,286 |
| Amazon-Movies | Amazon Review Data (2018), Movies and TV, ratings-only file `Movies_and_TV.csv` (8,765,568 ratings) | inherited 5-core filter, dense ids, first interaction per (user, item) | 297,377 | 59,925 |
| Douban-Movie | socialRec DoubanMovie dump (`Douban.tar.gz`, `movie/douban_movie.tsv`) | records from 2 January 2010 (local dates, UTC+8) onward, iterative 10-core, first interaction per (user, item) | 48,799 | 26,813 |

Every record is an implicit interaction: only the user, the item and the timestamp are used
and no rating threshold is applied, so records with low or missing ratings are interactions
too.  In the Douban-Movie source file, 22.6% of the records (2,652,182 of 11,742,260) carry
the rating -1, which the dump's README defines as an interaction without a rating; the share
is 18.9% after the date and 10-core filters, 21.6% of the training interactions and 15.9% /
14.8% of the validation / test targets.  The Amazon files have ratings 1-5 only.  Review text
and the Douban social network are not used.

`prep/adapters/` writes `data_raw/<dataset>/raw_interactions.csv` (tab-separated
`user item rating timestamp`, dense integer ids, unix seconds):

```bash
# Douban: download the socialRec dump, then
python prep/adapters/douban.py --tsv data_raw/Douban/movie/douban_movie.tsv
# Amazon Review Data (2018), https://nijianmo.github.io/amazon/ (Ni et al., 2019)
python prep/adapters/amazon.py --dataset Amazon-VG --src_reviews Video_Games_5.json.gz
python prep/adapters/amazon.py --dataset Amazon-Movies --src_reviews Movies_and_TV.csv
```

Amazon preprocessing, in order (`--src_reviews`):

1. Read (reviewerID, asin, rating, unix time) of every row; repeated reviews of the same
   (user, item) pair are still present at this stage.
2. The 5-core filter inherited from the code base the experiments started from: every
   round keeps the users with at least five rows on still-kept items and then the items
   with at least five rows among the users kept in the previous round, until a round
   removes nothing.  (It is not the usual simultaneous k-core; `--src_raw` applies that
   one and yields a different item set.)  Result: 496,904 rows / 55,144 users / 17,286
   items (VG) and 3,408,612 / 297,377 / 59,925 (Movies).
3. Dense ids = position of the original id in the sorted list of the kept reviewerID /
   asin strings (`data_raw/<dataset>/user_ids.csv`, `item_ids.csv`).
4. `prep/prep_split.py` keeps the first interaction per (user, item) and builds the split.

This reproduces the manuscript's interaction set and ids exactly (checked from the
ratings-only Movies file and, for Video Games, from a stored copy of the review records
with the 5-core file's 497,577 rows).  The manuscript's input files additionally carried an
arbitrary order of rows with equal timestamps, inherited from an earlier split of the same
data; the adapter writes rows in (timestamp, user, item) order instead.  The order affects
the byte content of the split files and the composition of mini-batches, not which
interactions are training, validation or test, the targets, the masks or any popularity
statistic.  `--src_files` merges already preprocessed dense-id files as they are.

```bash
python prep/verify_data.py --dataset Amazon-VG        # after prep_split.py; also Amazon-Movies, Douban-movie
```

compares your `data/<dataset>/` (and `data_raw/<dataset>/raw_interactions.csv`) with
`data/reference/checksums.json`: SHA-256 digests of the manuscript's files and
order-independent digests of their content (interaction sets, targets, masks,
training-block popularity tables, block edges).  "content same" on every line means the
split is the manuscript's.  `data/reference/<dataset>.json` records the metadata of the
splits (numbers of users, items, interactions, evaluation block).

## Chronological split (`prep/prep_split.py`)

```bash
python prep/prep_split.py --dataset Amazon-VG        # also Amazon-Movies, Douban-movie
```

1. Repeated (user, item) pairs keep their first interaction (first-interaction
   prediction; block counts are new-adoption counts).
2. The timeline is cut into fixed 60-day blocks starting at the first interaction.
3. The evaluation block t\* is the most active block among those preceded by at least
   half of all interactions.  This rule depends only on the data, not on any model.
4. Blocks are left-open, right-closed intervals (left edge, right edge] of the block grid
   (`np.searchsorted(..., side='left')` on the right edges), so an interaction at an edge
   belongs to the earlier block.  Inside t\*, interactions before the cut
   start + `--val_days` days (30 for 60-day blocks) form the validation window and the
   rest, up to and including the block's right edge, the test window, with boundaries
   shared by all users.  Training = every interaction before t\*; blocks after t\* are
   discarded.
5. Timestamps have daily resolution at local midnight (Amazon: UTC; Douban-Movie: UTC+8),
   and the block edges fall on midnights.  The edge date of the evaluation-block start is
   therefore the last training day, the validation window covers the 29 days after it, and
   the test window the remaining 31 days of the block.
6. Popularity statistics (cumulative counts, per-block counts, PDA tables, TIDE /
   CausalEPP interaction times) use training interactions only.

Boundaries of the splits used in the manuscript (unix seconds, with the local date of the
midnight they mark):

| Dataset | First interaction | Blocks before t\* | Evaluation block start | Validation / test cut | Evaluation block end |
| --- | --- | --- | --- | --- | --- |
| Amazon-VG | 939859200 (1999-10-14) | 92 | 1416787200 (2014-11-24) | 1419379200 (2014-12-24) | 1421971200 (2015-01-23) |
| Amazon-Movies | 881020800 (1997-12-02) | 105 | 1425340800 (2015-03-03) | 1427932800 (2015-04-02) | 1430524800 (2015-05-02) |
| Douban-Movie | 1262361600 (2010-01-02, UTC+8) | 30 | 1417881600 (2014-12-07, UTC+8) | 1420473600 (2015-01-06, UTC+8) | 1423065600 (2015-02-05, UTC+8) |

Training contains every interaction up to and including the evaluation block start;
validation covers (start, cut) and test [cut, end].  In calendar days:

| Dataset | Validation window | Test window |
| --- | --- | --- |
| Amazon-VG | 2014-11-25 to 2014-12-23 (29 days) | 2014-12-24 to 2015-01-23 (31 days) |
| Amazon-Movies | 2015-03-04 to 2015-04-01 (29 days) | 2015-04-02 to 2015-05-02 (31 days) |
| Douban-Movie | 2014-12-08 to 2015-01-05 (29 days, UTC+8) | 2015-01-06 to 2015-02-05 (31 days, UTC+8) |

`analysis/dataset_stats.py` computes these dates and day counts from `data/<dataset>/`
(columns `val_first_day` ... `test_calendar_days` of `dataset_table.csv`).

## Evaluation targets and candidates

* Validation and test users both need interactions in the training period: a user
  who first appears in the validation window is not a test user.  Targets are restricted
  to items with at least one training interaction and are unique per user (the split
  guarantees they are disjoint from the masked history).  Users without a remaining
  target are not evaluated.
* Every evaluated user is ranked against the full catalog; the observed history is
  masked (training history for validation; training history plus the user's filtered
  validation targets for test).  No candidate sampling.
* Model inputs (embeddings, graphs, popularity signals) are computed from training data
  for validation and test alike; only the mask grows with time.
* Serving-time popularity inputs differ by method and are kept as each method defines
  them; all of them are computed from training interactions only.
  STFR: the fresh signal of the last completed training block t\*-1, for validation and
  test alike.  STFR (forecast): the fresh signals of blocks t\*-2 and t\*-1, linearly
  extrapolated and clipped to [0, 1] (Eq. 8).  PDA: its per-block popularity of blocks
  t\*-2 and t\*-1, linearly extrapolated.  TPAB: the popularity group of the extrapolated
  count of blocks t\*-2 and t\*-1.  TIDE: the time-decayed popularity of the training interactions evaluated
  at the last training timestamp for validation and at the validation/test cut for test
  (the end of the input observed at test time).  CausalEPP: the same decayed popularity
  evaluated at the end of block t\*-1 for validation and at the validation/test cut for
  test; its item popularity and user sensitivity tables (six-month window ending with
  block t\*-1) are read at block t\*-1.

| Count | Amazon-VG val | Amazon-VG test | Amazon-Movies val | Amazon-Movies test | Douban-Movie val | Douban-Movie test |
| --- | --- | --- | --- | --- | --- | --- |
| Evaluation targets | 4,532 | 5,195 | 47,472 | 43,155 | 80,459 | 84,269 |
| Evaluated users | 2,381 | 2,555 | 22,718 | 21,342 | 10,920 | 10,947 |

Training-active catalog (items with c_i > 0, used by coverage and exposure Gini):
14,189 (Amazon-VG), 54,774 (Amazon-Movies), 23,425 (Douban-Movie).

## Temporal block configurations (Section 7.5)

```bash
python prep/prep_split.py --dataset Amazon-VG --block_days 15  --eval_block 368 --out_name Amazon-VG_b15
python prep/prep_split.py --dataset Amazon-VG --block_days 30  --eval_block 184 --out_name Amazon-VG_b30
python prep/prep_split.py --dataset Amazon-VG --block_days 120 --eval_block 46  --out_name Amazon-VG_b120
```

The main split's evaluation block starts 5520 days after the first interaction, so
`--eval_block 5520 / L` keeps the same training interactions (247,969) and the same
evaluation-block start for every block length; the block grid, the fresh signal and the
validation/test windows follow L.  With `--val_days` = L/2 (7, 15, 30 and 60 days), the
boundary rule above gives validation/test windows of 6/9, 14/16, 29/31 and 59/61 calendar
days for L = 15, 30, 60 and 120.
`scripts/run_block_configs.sh` runs the preprocessing and the experiments.

## Output files of `prep/prep_split.py`

| File | Content |
| --- | --- |
| `train_data.csv`, `val_data.csv`, `test_data.csv`, `trainval_data.csv` | interactions with `split_idx` (block index) |
| `train_list.txt`, `trainval_list.txt` | per-user chronological history (`uid n item...`), the masks |
| `val_list.txt`, `test_list.txt` | per-user targets (`uid item...`) |
| `all_times.npy` | right edges of the blocks |
| `item_frequency.csv` | cumulative training count per item (stale signal, nALRP, DICE / IPS popularity) |
| `item_frequency_all_times.csv` | per-block item counts (fresh signal; CausalEPP with `--cepp_window 1`) |
| `user_frequency_all_times.csv`, `user_frequency_high_popularity_all_times.csv` | per-block user counts (CausalEPP sensitivity with `--cepp_window 1`; the default six-month window tables are computed from `train_data.csv` when the model is built) |
| `PDA_popularity.csv` | PDA per-block popularity (Laplace smoothing, per-block min-max) |
| `DICE_popularity.npy` | normalized log cumulative popularity (DICE, IPS) |
| `item_interactions.csv` | training interaction times per item (TIDE / CausalEPP decayed popularity) |
| `dataset_meta.json` | sizes, block index of the evaluation block, boundaries |
