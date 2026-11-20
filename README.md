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
per refresh). Then, per trophy, the median is a saturating curve fitted to
those reads:

```
rate(t) = C * (1 - exp(-(t - d) / tau))
```

with `t` in hours since the release day began. A trophy's earn rate is the
share of tracked players who have reached it so far: it rises as players
play and saturates at the share who will ever reach it, so a saturating
curve is the shape the data has to have. `(tau, d)` are found on a grid and
`C` by least squares in rate space. With fewer than 4 distinct reads the
curve is unidentified and the forecast falls back to a (shrunk)
logit-linear trend, or to persistence.

The spread (logit sd) combines a random-walk term growing with the horizon,
the fit's residual, the trend's slope uncertainty, and a refresh-jump term:
the typical logit step between consecutive distinct reads once the fitted
curve is removed, times the square root of the refreshes expected before
the target. With no change ever seen (a flat launch night), a prior jump
size stands in, so a stale last read still earns a wide band — the next
refresh can move every rate at once.

A trophy at 100% in the last read stays (nearly) 100%; a trophy nobody has
earned yet keeps a small positive median. The program is deterministic:
two runs write byte-identical forecasts.

## Validation

On the kit's practice cases (launch night, history flat, answer the first
refresh), scored against the answer reconstructed from the kit's own facts
(3,868 players tracked, 4,276 trophies earned, trophy #2 still 100%),
this entry scores a mean pinball loss of about **1.29**, against about
**1.55** for the house persistence baseline — the wider, refresh-aware
spread is what earns the difference, since no trend is visible yet.

On four synthetic final-like cases (2.5 days of reads, refreshes every
2–4 h, player counts growing to hundreds of thousands, rates evolving on
saturating, lognormal, slow and dilution-dominated curves), it beats the
house baseline on every one: losses of 0.26 / 0.45 / 1.19 / 0.26 against
0.56 / 0.52 / 2.07 / 0.30.

Verified under a grader-like simulation: read-only repository copy, run
as uid 65534, empty output directory, one case as input — exit 0 in
~0.05 s, two runs byte-identical, output shape valid (55 rows, five
non-decreasing quantiles each, 0–100).

## License

MIT — see [LICENSE](LICENSE).
