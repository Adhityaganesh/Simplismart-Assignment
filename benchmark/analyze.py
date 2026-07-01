"""Load raw JSON results, compute stats, and generate charts."""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
CHARTS_DIR = Path(__file__).parent.parent / "results" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_latency():
    rows = []
    for f in RAW_DIR.glob("latency_*.json"):
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def load_concurrency():
    records = []
    for f in RAW_DIR.glob("concurrency_*.json"):
        data = json.loads(f.read_text())
        platform = list(data.values())[0]["requests"][0]["platform"]
        for c_str, v in data.items():
            total_tok = sum(r["output_tokens"] or 0 for r in v["requests"])
            records.append({
                "platform": platform,
                "concurrency": int(c_str),
                "wall_s": v["wall_s"],
                "throughput_tok_s": total_tok / v["wall_s"],
                "mean_ttft_s": sum(r["ttft_s"] or 0 for r in v["requests"]) / len(v["requests"]),
            })
    return pd.DataFrame(records)


def plot_latency(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, label in [
        ("ttft_s", axes[0], "TTFT (s)"),
        ("tpot_s", axes[1], "TPOT (s/token)"),
    ]:
        sns.boxplot(data=df, x="platform", y=metric, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
    fig.suptitle("Latency: Modal vs Simplismart")
    out = CHARTS_DIR / "latency_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_throughput(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for platform, grp in df.groupby("platform"):
        ax.plot(grp["concurrency"], grp["throughput_tok_s"], marker="o", label=platform)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.legend()
    out = CHARTS_DIR / "throughput_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    lat = load_latency()
    conc = load_concurrency()

    if not lat.empty:
        print("\n=== Latency summary ===")
        print(lat.groupby("platform")[["ttft_s", "tpot_s", "total_latency_s"]].describe())
        plot_latency(lat)

    if not conc.empty:
        print("\n=== Throughput summary ===")
        print(conc.to_string(index=False))
        plot_throughput(conc)


if __name__ == "__main__":
    main()
