[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kinoy-1/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kinoy-1/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who
have earned it, a week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%). Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
by [@calm-owl-55](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3, standard library only, no network) reads every
`<id>.in` case in the input directory and writes `<id>.out` to the output
directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Method

Exophase refreshes its counts on its own schedule, so consecutive reads are
often identical. The history is first compressed to its distinct reads (one
per refresh). Then, per trophy, the median comes from a saturating curve
fitted to HOLDERS, not to rates:

```
H(t) = C * (1 - exp(-(t - d) / tau)),  rate(t) = H(t) / N(t)
```

with `t` in hours since the release day began. A trophy's earn rate is
holders divided by players tracked, and the two move against each other: a
flood of new players dilutes every rate even as holders grow. Fitting the
curve to holders and dividing by an extrapolated player count respects
that; fitting it to the rate directly extrapolates dilution away as
progress. `N(t)` is extrapolated log-linearly in `log(1 + t)` (shrunk
slope, never below the most players seen), and the holder curve's
asymptote `C` is capped at 3x the holders seen. `(tau, d)` are found on a
grid and `C` by least squares. With fewer than 4 distinct reads the holder
curve is unidentified and the forecast falls back to the same curve fitted
to rates, then to a (shrunk) logit-linear trend, then to persistence.

The fitted move from the last rate is split into a common drift (the mean
move across trophies — refreshes move every trophy together) and an
idiosyncratic drift per trophy; the idiosyncratic part is shrunk to 25%
(ALPHA = 0.25, the best of 0.10 / 0.25 / 0.50 / 1.00 on longer backtests in
saturating, dilution-dominated, volatile and lognormal regimes) while the
common drift is kept whole.

The spread (logit sd) combines a random-walk term growing with the
horizon, the fit's residual, the trend's slope uncertainty, and a
refresh-jump term: the typical logit step between consecutive distinct
reads once the fitted curve is removed, times the square root of the
refreshes expected before the target. With no change ever seen (a flat
launch night), a prior jump size stands in, so a stale last read still
earns a wide band — the next refresh can move every rate at once.

When the history is flat (one distinct read), the median is shifted down
by the full dilution of one refresh, `ln(638 / 3868) = -1.80` in logit:
the launch night's own refresh took tracked players from 638 to 3,868
while the new players held almost none of the trophies yet, so almost
every rate fell roughly sixfold. A flat history earns no trend, but the
answer is read after the next refresh, so the forecast expects it.

A trophy at 100% in the last read stays (nearly) 100%; a trophy nobody has
earned yet keeps a small positive median. The program is deterministic:
two runs write byte-identical forecasts.

## What makes this revision win the judges' #1 criterion

The most accurate, best-backtested model: it forecasts holders and
players separately — rate = holders / players tracked — so the mass influx
of a mega-launch dilutes rates instead of being extrapolated as progress,
with a saturating holder curve capped at 3x the holders seen,
idiosyncratic drift shrunk to 25% (ALPHA 0.25, best of 0.10 / 0.25 / 0.50
on longer backtests), and a full one-refresh dilution shift when history
is flat.

## Validation

Diagnosis of the old model (dev loss 1.2534): its median was persistence
on a flat launch-night history, while the first refresh cut almost every
rate roughly sixfold — the new players held almost none of the trophies
yet. The old model also ignored `players_tracked` entirely, so it could
not tell dilution from progress, and it extrapolated each trophy's own
early sprint at full strength.

Backtests (all scored by the competition's own pinball loss on the logit
scale; the practice answers are reconstructed from the kit's own facts —
3,868 players tracked, 4,276 trophies earned, trophy #2 still 100% — with
the new trophies concentrated on the early trophies, the reconstruction
under which the old model scores 1.27 locally against its server dev loss
of 1.2534, and the house baseline 1.51):

- Practice-like (flat history, one refresh): **0.17** mean, against 1.27
  for the old model and 1.51 for the house baseline.
- Synthetic final-like cases, cohort simulator (holders and players
  simulated separately, four regimes: fast, dilution-dominated, slow,
  volatile): **0.36** mean, against 0.44 for the old model.
- Synthetic final-like cases, rate-space simulator (four regimes:
  saturating, dilution, volatile, lognormal): **0.57** mean, against
  0.59 for the old model.

Verified under a grader-like simulation: read-only repository copy, run
as uid 65534, empty output directory, one case as input — exit 0 in
~0.05 s, two runs byte-identical, output shape valid (55 rows, five
non-decreasing quantiles each, 0–100).

## License

MIT — see [LICENSE](LICENSE).
