import argparse, math, re, statistics
from pathlib import Path
ROW_RE = re.compile(r"\|\s*(?P<run>\S+)\s*\|\s*eval\s*\|\s*(?P<train>[0-9.]+)\s*\|\s*(?P<val>[0-9.]+)\s*\|\s*(?P<tta>[0-9.]+)\s*\|\s*(?P<time>[0-9.]+)\s*\|")
def sf(z): return 0.5 * math.erfc(z / math.sqrt(2))
ap=argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--target", type=float, required=True); a=ap.parse_args()
rows=[]
for line in Path(a.log).read_text(errors="replace").splitlines():
    m=ROW_RE.search(line)
    if m and m.group("run") != "warmup": rows.append({k: float(m.group(k)) for k in ("train","val","tta","time")})
if len(rows) < 2: raise SystemExit("need at least two real rows")
vals=[r["val"] for r in rows]; times=[r["time"] for r in rows]
n=len(vals); mean=statistics.fmean(vals); sd=statistics.stdev(vals); se=sd/math.sqrt(n); z=(mean-a.target)/se if se else float("inf")
print(f"runs={n}")
print(f"val_acc mean={mean:.6f} sd={sd:.6f} se={se:.6f} target={a.target:.6f} normal_one_sided_p_mean_le_target={sf(z):.6g}")
print(f"time_s mean={statistics.fmean(times):.6f} sd={statistics.stdev(times):.6f} min={min(times):.6f} max={max(times):.6f}")
print(f"hits={sum(v >= a.target for v in vals)}/{n}")
