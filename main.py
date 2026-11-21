"""Forecast GTA VI trophy earn rates on Exophase, one case at a time.

Reads every <id>.in in the input directory (schema
"makefun/gta6-trophies-case/v1") and writes <id>.out to the output directory:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, five
quantiles of each trophy's earn rate (percent) at the case's target.

Method, per trophy, from the case's own history only (standard library,
deterministic, no network):
  * Exophase refreshes its counts on its own schedule, so consecutive reads
    are often identical. The history is compressed to its distinct reads
    (one per refresh), each at its first appearance; a flat stretch is one
    observation, not many.
  * The median is a saturating curve fitted to HOLDERS, not to rates:
    a trophy's earn rate is holders / players-tracked, and the two move
    against each other -- a flood of new players dilutes every rate even
    as holders grow. So per trophy the curve rate(t) = C * (1 -
    exp(-(t - d) / tau)) is fitted to H(t) = rate(t) * N(t), t in hours
    since the release day began (2026-11-19T00:00:00Z), with C capped at
    3x the holders seen (holders can grow past what was seen, but the cap
    keeps the dilution prior the launch night showed: the first refresh
    took tracked players 638 -> 3,868 while new players held far fewer
    trophies each, so rates fell almost everywhere). N(t) is extrapolated
    log-linearly in log(1 + t) (shrunk slope), and the median rate at the
    target is H(target) / N(target). (tau, d) are found on a grid and C by
    least squares -- fitting in logit space was tried and is worse,
    because the page prints 2 decimals and the logit of a 0.01% rate is
    mostly rounding noise. With fewer than 4 distinct reads the holder
    curve is unidentified and the median falls back to the same curve
    fitted to rates, then to a logit-linear trend (shrunk toward
    persistence), then to persistence.
  * The fitted move from the last rate is split into a common drift (the
    mean move across trophies -- refreshes move every trophy together)
    and an idiosyncratic drift per trophy; the idiosyncratic part is
    shrunk to 25% (ALPHA = 0.25, the best of 0.10 / 0.25 / 0.50 / 1.00 on
    longer backtests in saturating, dilution-dominated, volatile and
    lognormal regimes) while the common drift is kept whole: extrapolating
    one trophy's own early sprint 4.5 days out is what the backtests
    punish most.
  * When the history is flat (a single distinct read: no trend is
    identified at all), the median is shifted down by the full dilution
    of one refresh, ln(638 / 3868) = -1.80 in logit -- the launch night's
    own refresh, which cut almost every rate roughly sixfold because the
    new players held almost none of the trophies yet. A flat history
    earns no trend, but the answer is read after the next refresh, so the
    forecast expects it.
  * Spread (logit sd) combines a random-walk term growing with the
    horizon, the saturating fit's residual, the trend's slope uncertainty,
    and a refresh-jump term: the typical logit step between consecutive
    distinct reads once the fitted curve is removed, times the square root
    of the refreshes expected before the target. With no change ever seen
    (a flat launch night), a prior jump size and gap stand in, so a stale
    last read still earns a wide band: the next refresh can move every
    rate at once.

Edge cases: a trophy at 100% in the last read stays (nearly) 100% with a
fixed narrow-down row; a trophy never earned yet keeps a small positive
median; missing rates (null) are skipped; a case with too little history
falls back to persistence with a wide spread.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0),
Python 3.13, standard library only, no network. -P keeps this file's
directory OFF sys.path; the sys.path line below puts it back first.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day began 2026-11-19T00:00:00Z; time is hours since then.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Idiosyncratic drift kept (ALPHA); the common drift is kept whole.
ALPHA_IDIO = 0.25
# Holder-curve asymptote cap, in units of the most holders seen.
HOLDER_CMAX = 3.0
# Full one-refresh dilution shift (logit) when the history is flat:
# ln(638 / 3868), the launch night's first refresh (players tracked
# 638 -> 3,868; almost every rate fell roughly sixfold).
DILUTION_SHIFT = -1.80
# Spread: logit sd per square-root day of drift uncertainty.
SIGMA_RW = 0.30
# Prior for a refresh jump when the history shows none (flat launch night).
JUMP_PRIOR_SD = 0.8
JUMP_PRIOR_GAP_H = 4.0
# Spread floor and cap, in logit units; refresh term capped separately.
SIGMA_MIN = 0.18
SIGMA_MAX = 1.2
SIGMA_REFRESH_MAX = 0.6
# Largest |slope| allowed for the fallback trend, logit per ln(1 + hours).
SLOPE_CAP = 2.5
# Grid for the saturating curve: time constant tau (hours) and delay d.
TAUS = [6, 12, 24, 48, 96, 192, 384, 768, 1536]
DELAYS = [0, 6, 12, 24, 48, 72]
# Distinct reads needed before the saturating curve is trusted.
MIN_POINTS_SAT = 4
# A trophy at 100% in the last read: its fixed row.
FULL_ROW = [98.5, 99.7, 100.0, 100.0, 100.0]
# A trophy nobody has earned yet: small positive median, wide spread.
ZERO_MEDIAN = 0.05
ZERO_SIGMA = 1.0


def parse_time(text):
    """ISO 8601 (date or datetime, Z or offset) -> aware datetime (UTC)."""
    text = text.strip()
    if len(text) == 10:  # a bare date: midnight UTC that day
        text += "T00:00:00Z"
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
        timezone.utc)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def wls(xs, ys, ws=None):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_sd, se_b). With fewer than 3 points the slope is 0
    and the intercept is the (weighted) mean; with a degenerate x-spread the
    slope is 0 too.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    s = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    if n < 3:
        return sy / s, 0.0, 0.0, 0.0
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    den = s * sxx - sx * sx
    if den <= 1e-12:
        return sy / s, 0.0, 0.0, 0.0
    b = (s * sxy - sx * sy) / den
    a = (sy - b * sx) / s
    dof = n - 2
    var = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    var = max(var, 0.0)
    se_b = math.sqrt(var / (sxx - sx * sx / s)) if sxx - sx * sx / s > 0 else 0.0
    return a, b, math.sqrt(var), se_b


def shrink_slope(b, n):
    """Shrink a fitted slope toward 0 and cap it: short histories persist."""
    if n < 3:
        return 0.0
    b = b * n / (n + 2.0)
    return max(-SLOPE_CAP, min(SLOPE_CAP, b))


def distinct_reads(history):
    """One entry per refresh: drop reads identical to the previous kept one."""
    kept = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev:
            kept.append(read)
            prev = key
    return kept


def fit_saturating(ts, ys, cmax=1.0):
    """Fit rate = C * (1 - exp(-(t - d) / tau)) by grid over (tau, d).

    ts in hours since release, ys in fractions (0..1). C by least squares
    in rate space, at least the largest value seen, at most cmax (1.0 for
    a rate, which cannot exceed 1; larger for holders, whose asymptote
    can exceed the holders seen so far). Returns (predict_fn,
    resid_sd_logit) or None when the fit is degenerate.
    """
    best = None
    for tau in TAUS:
        for d in DELAYS:
            u = [1.0 - math.exp(-max(0.0, t - d) / tau) for t in ts]
            uu = sum(v * v for v in u)
            if uu <= 1e-12:
                continue
            c = sum(a * b for a, b in zip(u, ys)) / uu
            c = max(max(ys), min(cmax, c))
            sse = sum((y - c * v) ** 2 for y, v in zip(ys, u))
            if best is None or sse < best[0]:
                best = (sse, c, tau, d)
    if best is None:
        return None
    _, c, tau, d = best

    def predict(t):
        return c * (1.0 - math.exp(-max(0.0, t - d) / tau))

    # Residual sd in logit space (what the score is measured on).
    res = []
    for t, y in zip(ts, ys):
        f = predict(t)
        res.append(logit(max(y, 1e-4) * 100.0)
                   - logit(max(f, 1e-4) * 100.0))
    n = len(res)
    sd = math.sqrt(sum(r * r for r in res) / max(1, n - 3)) if n > 3 else 0.0
    return predict, sd


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    # Time anchor: the release date the case names, else release day.
    try:
        anchor = parse_time(str(case.get("game", {}).get("release_date",
                                                       "2026-11-19")))
    except ValueError:
        anchor = RELEASE
    try:
        target = parse_time(case["target"])
    except (KeyError, ValueError):
        target = parse_time(case["at"]) + timedelta(
            minutes=case["horizon_minutes"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    t_target = max(0.0, (target - anchor).total_seconds() / 3600.0)
    x_target = math.log1p(t_target)

    reads = distinct_reads(history)
    ts = []
    for read in reads:
        t = max(0.0, (parse_time(read["at"]) - anchor).total_seconds() / 3600.0)
        ts.append(t)
    xs = [math.log1p(t) for t in ts]
    n_reads = len(reads)

    # Median gap between distinct reads, in hours (for the refresh term).
    gap_hours = []
    for j in range(1, n_reads):
        gap_hours.append((parse_time(reads[j]["at"])
                          - parse_time(reads[j - 1]["at"])).total_seconds()
                         / 3600.0)
    gap_hours.sort()
    gap_h = (gap_hours[len(gap_hours) // 2] if gap_hours
             else JUMP_PRIOR_GAP_H)
    gap_h = max(gap_h, 0.25)
    n_refreshes = max(horizon_h / gap_h, 0.0)
    sigma_rw = SIGMA_RW * math.sqrt(horizon_d)

    # Extrapolate the players tracked: log N ~ a + b*log(1 + t), slope
    # shrunk; tracked players only grow, so never predict below the most
    # seen. (The holder forecast divides by this at the target.)
    n_obs = [max(1.0, float(r.get("players_tracked") or 1)) for r in reads]
    a_n, b_n, _, _ = wls(xs, [math.log(v) for v in n_obs])
    b_n = shrink_slope(b_n, len(xs))
    n_target = max(max(n_obs), math.exp(a_n + b_n * x_target))

    # Per-trophy fits first (the refresh term needs the fitted curves).
    fits = []
    for i in range(len(trophies)):
        pts = [(ts[j], reads[j]["rates"][i],
                max(1.0, float(reads[j].get("players_tracked") or 1)))
               for j in range(n_reads)
               if reads[j]["rates"][i] is not None]
        tt = [p[0] for p in pts]
        yy = [p[1] for p in pts]
        xx = [math.log1p(t) for t in tt]
        ly = [logit(y) for y in yy]
        a_lin, b_lin, _, se_b = wls(xx, ly)
        b_lin = shrink_slope(b_lin, len(xx))
        med_lin = inv_logit(a_lin + b_lin * x_target)
        x_bar = sum(xx) / len(xx) if xx else 0.0
        sat = None
        if len(tt) >= MIN_POINTS_SAT and max(yy) > 0:
            sat = fit_saturating(tt, [y / 100.0 for y in yy])
        # Holder curve: H(t) = rate(t) * N(t), fitted on the same
        # saturating shape scaled by the most holders seen, its asymptote
        # allowed up to HOLDER_CMAX times that (holders grow past what
        # was seen; the cap keeps the launch night's dilution prior).
        sat_h = None
        if len(tt) >= MIN_POINTS_SAT and max(yy) > 0:
            hs = [p[1] / 100.0 * p[2] for p in pts]
            max_h = max(hs)
            if max_h > 0:
                fitted_h = fit_saturating(
                    tt, [h / max_h for h in hs], cmax=HOLDER_CMAX)
                if fitted_h is not None:
                    sat_h = (fitted_h, max_h)
        fits.append({"tt": tt, "yy": yy, "xx": xx, "ly": ly,
                     "med_lin": med_lin, "se_b": se_b, "x_bar": x_bar,
                     "sat": sat, "sat_h": sat_h})

    # Refresh-jump sd: pooled logit steps between consecutive distinct reads
    # after removing each trophy's fitted curve (saturating where fitted,
    # else the linear trend). No change ever seen -> a prior stands in.
    jumps = []
    for i, f in enumerate(fits):
        tt, yy = f["tt"], f["yy"]
        if len(tt) < 2:
            continue
        if f["sat"] is not None:
            pred, _ = f["sat"]
            # logit of the fitted fraction, computed directly:
            curve = []
            for t in tt:
                p = max(min(pred(t), 0.9999), 0.0001)
                curve.append(math.log(p / (1 - p)))
        else:
            a_lin, b_lin, _, _ = wls(f["xx"], f["ly"])
            b_lin = shrink_slope(b_lin, len(tt))
            curve = [a_lin + b_lin * x for x in f["xx"]]
        for j in range(1, len(tt)):
            jumps.append((f["ly"][j] - curve[j])
                         - (f["ly"][j - 1] - curve[j - 1]))
    if jumps:
        mean_jump = sum(jumps) / len(jumps)
        jump_sd = math.sqrt(sum((d - mean_jump) ** 2 for d in jumps)
                            / max(1, len(jumps) - 1))
    else:
        jump_sd = JUMP_PRIOR_SD
        gap_h = JUMP_PRIOR_GAP_H
        n_refreshes = max(horizon_h / gap_h, 0.0)
    sigma_refresh = min(SIGMA_REFRESH_MAX,
                        jump_sd * math.sqrt(n_refreshes))

    # Model median per trophy (holder curve first, then the rate curve,
    # then the trend), and its drift from the last rate, in logit space.
    # The drift splits into a common part (refreshes move every trophy
    # together) and an idiosyncratic part, shrunk to ALPHA_IDIO.
    model_logits = []
    drifts = []
    last_rates = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            last_rate = 0.0
        last_rates.append(last_rate)
        if not f["yy"] or max(f["yy"]) <= 0 or last_rate <= 0:
            model_logits.append(None)
            drifts.append(0.0)
            continue
        if f["sat_h"] is not None:
            (pred_h, _), max_h = f["sat_h"]
            h_target = pred_h(t_target) * max_h
            med = max(0.01, min(99.99, h_target / n_target * 100.0))
            model_logits.append(logit(med))
        elif f["sat"] is not None:
            pred, _ = f["sat"]
            med = max(0.01, min(99.99, pred(t_target) * 100.0))
            model_logits.append(logit(med))
        else:
            model_logits.append(logit(f["med_lin"]))
        drifts.append(model_logits[-1] - logit(max(last_rate, 0.01)))
    common_drift = (sum(drifts) / len(drifts)) if drifts else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        last_rate = last_rates[i]

        if last_rate >= 99.995:
            # Everybody holds it, through a sixfold influx on launch night:
            # it stays (nearly) 100%, with a small allowance for a dip.
            rows.append({"id": trophy["id"], "q": list(FULL_ROW)})
            continue

        if not f["yy"] or max(f["yy"]) <= 0:
            # Nobody has earned it yet: a small positive median, wide band.
            med_logit = logit(ZERO_MEDIAN)
            sigma = ZERO_SIGMA
        else:
            if n_reads <= 1:
                # Flat history: no trend is identified, but the answer is
                # read after the next refresh, so expect one full refresh
                # of dilution -- the launch night's own, ln(638 / 3868).
                med_logit = logit(max(last_rate, 0.01)) + DILUTION_SHIFT
                resid = 0.0
            else:
                idio = drifts[i] - common_drift
                med_logit = (logit(max(last_rate, 0.01))
                             + ALPHA_IDIO * idio + common_drift)
                resid = f["sat"][1] if f["sat"] is not None else 0.0
            sigma_slope = f["se_b"] * abs(x_target - f["x_bar"])
            sigma = math.sqrt(sigma_rw ** 2 + resid ** 2
                              + sigma_slope ** 2 + sigma_refresh ** 2)
            sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))

        qs = []
        prev_q = None
        for z in Z:
            q = round4(clamp(inv_logit(med_logit + z * sigma)))
            if prev_q is not None and q < prev_q:
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
