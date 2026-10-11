"""Rank a making benchmark's results (scripts/making-bench/making-bench).

    python3 scripts/making-bench/summarize.py OUT [OUT ...] [--worst N] [--base OUT]

Each OUT is a `making-bench --out` directory (one host, one sha). Prints, per OUT: the host, the
makings' total, p50, p95 and max wall milliseconds, the worst N makings (20 by default) with their
largest module's SPIR-V words, and per state its runs: creation and first-draw milliseconds, the
creation's translation and setup outside the driver (`created_ms` less the makings inside it),
and the GPU milliseconds of one draw (median and minimum); for a run made with `--queue-first`,
the queue's own creation before the state's too. With `--base OUT`, each
state's makings are also compared with that directory's (the same set at another sha). Times,
counts and words only."""
import csv
import os
import sys


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def percentile(values, p):
    """The nearest-rank percentile `p` (0-100) of `values`, sorted."""
    if not values:
        return None
    k = max(0, min(len(values) - 1, int(round(p / 100 * len(values) + 0.5)) - 1))
    return values[k]


def fmt(v, digits=1):
    return "-" if v is None else f"{v:,.{digits}f}"


def by_state(makings):
    out = {}
    for m in makings:
        out.setdefault(m["id"], []).append(float(m["ms"]))
    return out


def summarize(out, worst, base):
    host = open(os.path.join(out, "host.txt")).read().strip().splitlines()
    makings = rows(os.path.join(out, "makings.tsv"))
    runs = rows(os.path.join(out, "runs.tsv"))
    print(f"## {out}")
    for line in host:
        print(f"    {line.strip()}")
    ms = sorted(float(m["ms"]) for m in makings)
    total = sum(ms)
    print(
        f"makings {len(ms)}: total {fmt(total / 1000, 2)} s, p50 {fmt(percentile(ms, 50))} ms, "
        f"p95 {fmt(percentile(ms, 95))} ms, max {fmt(ms[-1] if ms else None)} ms"
    )
    failed = [r for r in runs if r["rc"] != "0"]
    print(f"runs {len(runs)}, failed {len(failed)}")
    for r in failed:
        print(f"    FAILED {r['id']} {r['combo']} rc {r['rc']}")
    print(f"\nworst {worst} makings:")
    print("| # | state | combo | pipeline | at | ms | words |")
    print("|---|---|---|---|---|---|---|")
    ranked = sorted(makings, key=lambda m: -float(m["ms"]))[:worst]
    for i, m in enumerate(ranked, 1):
        print(
            f"| {i} | {m['id']} | {m['combo']} | {m['pipeline']} | {m['at']} | "
            f"{fmt(float(m['ms']))} | {int(m['words']):,} |"
        )
    # A queue-first run's queue creation, apart.
    queue_first = any("queue_first_ms" in r for r in runs)
    print("\nper run:")
    print(
        "| state | combo | created ms | translation+setup ms | first draw ms | GPU ms median/min | maxrss MB |"
        + (" queue first ms |" if queue_first else "")
    )
    print("|---|---|---|---|---|---|---|" + ("---|" if queue_first else ""))
    for r in runs:
        created, making = num(r["created_ms"]), num(r["creation_making_ms"])
        setup = None if created is None or making is None else created - making
        rss = num(r["maxrss_kb"])
        print(
            f"| {r['id']} | {r['combo']} | {fmt(created)} | {fmt(setup)} | "
            f"{fmt(num(r['first_draw_ms']))} | {fmt(num(r['gpu_ms_median']), 3)}/"
            f"{fmt(num(r['gpu_ms_min']), 3)} | {fmt(None if rss is None else rss / 1024, 0)} |"
            + (f" {fmt(num(r.get('queue_first_ms')))} |" if queue_first else "")
        )
    if base:
        then = by_state(rows(os.path.join(base, "makings.tsv")))
        now = by_state(makings)
        print(f"\nagainst {base} (sum of each state's makings, ms):")
        print("| state | base | this | ratio |")
        print("|---|---|---|---|")
        for ident in sorted(set(then) | set(now)):
            a, b = sum(then.get(ident, [])), sum(now.get(ident, []))
            ratio = f"{b / a:.2f}x" if a else "-"
            print(f"| {ident} | {fmt(a)} | {fmt(b)} | {ratio} |")
    print()


def main(argv):
    worst, base, outs = 20, None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--worst":
            worst = int(argv[i + 1])
            i += 2
        elif argv[i] == "--base":
            base = argv[i + 1]
            i += 2
        else:
            outs.append(argv[i])
            i += 1
    if not outs:
        sys.exit(__doc__)
    for out in outs:
        summarize(out, worst, base)


if __name__ == "__main__":
    main(sys.argv[1:])
