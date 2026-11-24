
import math, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SEED_FIT, SEED_TIMING = 7, 3    
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED = "#2a6f97", "#d9822b", "#b03a2e"

def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

def ztest(k1, n1, k2, n2):
    """Two-proportion z statistic for k1/n1 minus k2/n2 (pooled variance)."""
    p = (k1 + k2) / (n1 + n2)
    return (k1/n1 - k2/n2) / math.sqrt(p*(1-p)*(1/n1 + 1/n2))

pct = lambda x: f"{100*x:.1f}%"

# --- By inning and challenges remaining: (inning, chal, overturned) with 2 left, then with 1 left
INNING = [(1,995,605,37,24),(2,878,548,59,39),(3,903,531,144,87),(4,866,477,184,116),
          (5,849,485,276,145),(6,885,472,349,193),(7,829,469,447,246),(8,724,319,556,283),
          (9,685,250,681,303),(10,34,12,116,52),(11,10,3,31,18),(12,2,1,16,7),(13,0,0,1,0)]
inn = pd.DataFrame(INNING, columns=["inn","n2","k2","n1","k1"]).set_index("inn")
reg = inn.loc[1:9]                # regulation innings; the model covers these only

# --- By role
ROLE = pd.DataFrame({"chal":[5609,4766,182], "ot":[3285,2328,72]},
                    index=["catcher","batter","pitcher"])

# --- By count: balls, strikes, all (chal, ot%), as batter (chal, ot%), defense (chal, ot%)
COUNT = pd.DataFrame([
 (0,0,2417,59.8, 933,55.4,1484,62.6),(0,1,1065,60.8, 449,49.2, 616,69.2),(0,2, 570,51.8, 246,41.1, 324,59.9),
 (1,0, 953,60.4, 436,63.5, 517,57.8),(1,1,1041,54.6, 518,48.1, 523,61.0),(1,2, 884,48.0, 410,40.2, 474,54.6),
 (2,0, 419,58.0, 192,63.5, 227,53.3),(2,1, 678,51.6, 346,46.8, 332,56.6),(2,2, 968,46.7, 468,40.6, 500,52.4),
 (3,0, 137,54.7,  57,57.9,  80,52.5),(3,1, 438,46.6, 216,46.8, 222,46.4),(3,2, 987,41.0, 495,38.4, 492,43.7)],
 columns=["balls","strikes","n_all","pct_all","n_bat","pct_bat","n_def","pct_def"])

# --- By leverage index (page buckets)
LEV = pd.DataFrame([("0.0-0.5 very low",3193,52.4),("0.5-1.0 low",3066,57.7),("1.0-1.5 medium",1806,54.0),
                    ("1.5-2.5 high",1615,53.4),("2.5+ very high",877,46.2)],
                   columns=["bucket","chal","pct"])
LEV["ot"] = (LEV.chal * LEV.pct / 100).round()     # rounded from the page's percentages

# --- By baserunners
BASES = pd.DataFrame([("Bases empty",5531,55.9),("Runner on 1st",2061,53.4),("Runner on 2nd",850,52.4),
                      ("Runner on 3rd",237,49.4),("Runners on 1st & 2nd",896,50.7),("Runners on 1st & 3rd",319,50.2),
                      ("Runners on 2nd & 3rd",260,53.1),("Bases loaded",403,43.9)], columns=["state","chal","pct"])

# --- Challenges per team-game
DIST = pd.Series({0:402,1:1147,2:1592,3:999,4:472,5:167,6:55,7:18,8:3,9:3})

# --- ESPN season rankings (hand-entered): Nationals catcher Drew Millas, challenges won of attempted
MILLAS = (22, 51)

# --- Sanity checks: every table must reproduce the same totals
TOTAL = int(ROLE.chal.sum())
assert TOTAL == 10557 and ROLE.ot.sum() == 5685
assert inn.n2.sum() + inn.n1.sum() == TOTAL and inn.k2.sum() + inn.k1.sum() == 5685
assert COUNT.n_all.sum() == TOTAL and COUNT.n_bat.sum() == ROLE.chal["batter"]
assert COUNT.n_def.sum() == ROLE.chal["catcher"] + ROLE.chal["pitcher"]
assert LEV.chal.sum() == TOTAL and BASES.chal.sum() == TOTAL
TEAM_GAMES = int(DIST.sum())
print(f"All tables agree: {TOTAL:,} challenges, {int(ROLE.ot.sum()):,} overturned "
      f"({pct(ROLE.ot.sum()/TOTAL)}), {TEAM_GAMES:,} team-games.")

# ---- derived quantities shared by several notebooks
early, late = reg.loc[1:3], reg.loc[8:9]
ke, ne = early.k2.sum(), early.n2.sum()
kl, nl = late.k2.sum(), late.n2.sum()
Cn, Ck = ROLE.loc["catcher"]; Bn, Bk = ROLE.loc["batter"]; Pn, Pk = ROLE.loc["pitcher"]
Dn, Dk = Cn + Pn, Ck + Pk
GAP_OBS = Dk/Dn - Bk/Bn
RUNS = [0.08, 0.12, 0.16]               # ASSUMED runs per overturned call
TEAMS, RUNS_PER_WIN = 30, 10            # ASSUMED conversion
reg_chal = int(reg.n2.sum() + reg.n1.sum()); reg_ok = int(reg.k2.sum() + reg.k1.sum())

S, T, NQ, KAPPA = 6, 54, 200, 4.0       # opportunities per inning, total, q grid, Beta concentration
QS = (np.arange(NQ) + .5) / NQ
SHARE_B = 0.451                         # observed batter share of challenges (4,766 / 10,557)

def q_weights(mu):
    a, b = mu*KAPPA, (1-mu)*KAPPA
    w = QS**(a-1) * (1-QS)**(b-1)
    return w / w.sum()

def stakes(t, m):
    """Stakes of a flipped call with t opportunities left: 1 in inning 1, rising linearly to m in inning 9."""
    return 1 + (m-1) * ((T-t)//S) / 8

def solve(r, mu, m=1.0):
    """Backward induction. Returns W (value) and D (option value of one challenge), indexed [k, t]."""
    w = q_weights(mu)
    W = np.zeros((3, T+1)); D = np.zeros((3, T+1))
    for t in range(1, T+1):
        v = stakes(t, m)
        W[0, t] = W[0, t-1]
        for k in (1, 2):
            d = W[k, t-1] - W[k-1, t-1]; D[k, t] = d
            th = d / (v + d)
            use = QS*(v + W[k, t-1]) + (1-QS)*W[k-1, t-1]
            W[k, t] = (1-r)*W[k, t-1] + r*(w*np.where(QS >= th, use, W[k, t-1])).sum()
    return W, D

def simulate(r, mu, m=1.0, phi_b=1.0, N=60_000, rng=None, draw_b=True):
    """Simulate N games. Returns one record per challenge made plus per-game challenge counts.
    Random draws are made in a fixed order (arrival, q, batter flag, outcome) so a seeded stream replays exactly."""
    rng = rng if rng is not None else np.random.default_rng(SEED_FIT)
    cum = np.cumsum(q_weights(mu)); cum /= cum[-1]; W, D = solve(r, mu, m)
    k = np.full(N, 2); n_game = np.zeros(N, int); ev = {n: [] for n in ("inn","k","ok","is_b","v")}
    for s in range(T):
        t, i = T - s, s//S + 1; v = stakes(t, m)
        arrive = rng.random(N) < r
        q = QS[np.minimum(np.searchsorted(cum, rng.random(N), side="right"), NQ-1)]
        is_b = rng.random(N) < SHARE_B if draw_b else np.zeros(N, bool)
        d = D[k, t]; f = np.where(is_b, phi_b, 1.0)
        ch = arrive & (k > 0) & (q >= f*d/(v + f*d))
        ok = ch & (rng.random(N) < q)
        n_game += ch
        for name, arr in (("inn", np.full(ch.sum(), i)), ("k", k[ch]), ("ok", ok[ch]),
                          ("is_b", is_b[ch]), ("v", np.full(ch.sum(), v))):
            ev[name].append(arr)
        k = k - (ch & ~ok)
    out = {n: np.concatenate(a) for n, a in ev.items()}
    out["n_game"], out["N"] = n_game, N
    return out

def summarize(s):
    return dict(chal_per_game=len(s["ok"])/s["N"], rate=s["ok"].mean(), zero=(s["n_game"] == 0).mean())

# Calibration targets, as stated in the paper: challenges per team-game, overall overturn rate, share of team-games with none
TGT = dict(chal_per_game=2.18, rate=0.539, zero=0.083)
_CACHE = {}

def game_model_run():
    """Replay the paper's main simulation stream (seed 7) in its original order: the grid fit, the base run,
    the batter-externality runs (phi = 1, 1/9, 0), the run behind Figure 2, then extra phi values."""
    if "game" in _CACHE: return _CACHE["game"]
    rng = np.random.default_rng(SEED_FIT); grid = []
    for r, mu in itertools.product([.03,.04,.05,.06,.08,.10], [.35,.45,.55,.65,.75]):
        m = summarize(simulate(r, mu, N=20_000, rng=rng))
        err = ((m["chal_per_game"]-TGT["chal_per_game"])/TGT["chal_per_game"])**2 \
            + ((m["rate"]-TGT["rate"])/TGT["rate"])**2 + ((m["zero"]-TGT["zero"])/.05)**2
        grid.append(dict(r=r, mu=mu, err=err, **m))
    grid = pd.DataFrame(grid).sort_values("err")
    r, mu = float(grid.iloc[0].r), float(grid.iloc[0].mu)
    run = dict(grid=grid, r=r, mu=mu, base=simulate(r, mu, N=60_000, rng=rng))
    for phi in (1.0, 1/9, 0.0): run[phi] = simulate(r, mu, phi_b=phi, N=60_000, rng=rng)
    run["fig"] = simulate(r, mu, N=60_000, rng=rng)
    for phi in (0.5, 0.25): run[phi] = simulate(r, mu, phi_b=phi, N=60_000, rng=rng)
    _CACHE["game"] = run
    return run

def timing_run(ms=(1.0, 1.5, 2.0, 3.0), r=0.05, mu=0.45, N=80_000):
    """Replay the timing extension's own stream (seed 3): one run per stakes multiplier, no batter draw."""
    if "timing" in _CACHE: return _CACHE["timing"]
    rng = np.random.default_rng(SEED_TIMING)
    _CACHE["timing"] = {m: simulate(r, mu, m=m, N=N, rng=rng, draw_b=False) for m in ms}
    return _CACHE["timing"]


# ---- fitted model values (notebook 09 re-runs the grid fit and checks that it still gives these)
R_FIT, MU_FIT = 0.05, 0.45
W1, D1 = solve(R_FIT, MU_FIT, 1.0)
start = [T - (i-1)*S for i in range(1, 10)]     # opportunities left at the start of each inning
