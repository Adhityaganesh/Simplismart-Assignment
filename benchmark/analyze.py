"""Load raw JSON results, compute stats, and generate charts."""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
CHARTS_DIR = Path(__file__).parent.parent / "results" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"modal": "#6C63FF", "simplismart": "#FF6B6B"}

# GPU hourly rates used for cost/token estimates.
# Modal L4: publicly listed at ~$0.80/hr.
# Simplismart A10: no public listing; using $0.75/hr as a mid-point of the
# ~$0.60-0.90/hr market range for A10 instances (assumption, not vendor-quoted).
HOURLY_RATE = {"modal": 0.80, "simplismart": 0.75}


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
    return pd.DataFrame(records).sort_values(["platform", "concurrency"]).reset_index(drop=True)


def load_coldstart():
    rows = []
    for f in RAW_DIR.glob("coldstart_*.json"):
        rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def load_input_sweep():
    rows = []
    for f in RAW_DIR.glob("input_sweep_*.json"):
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def load_output_sweep():
    rows = []
    for f in RAW_DIR.glob("output_sweep_*.json"):
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def plot_latency(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, label in [
        ("ttft_s", axes[0], "TTFT (s)"),
        ("tpot_s", axes[1], "TPOT (s/token)"),
    ]:
        sns.boxplot(data=df, x="platform", y=metric, ax=ax, hue="platform",
                    palette=COLORS, legend=False)
        ax.set_title(label)
        ax.set_xlabel("")
    fig.suptitle("Single-Stream Latency: Modal vs Simplismart")
    out = CHARTS_DIR / "latency_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_throughput(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for platform, grp in df.groupby("platform"):
        ax.plot(grp["concurrency"], grp["throughput_tok_s"], marker="o",
                label=platform, color=COLORS.get(platform))
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency: Modal vs Simplismart")
    ax.legend()
    out = CHARTS_DIR / "throughput_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coldstart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    df = df.sort_values("platform")
    bars = ax.bar(df["platform"], df["ttft_s"],
                   color=[COLORS.get(p) for p in df["platform"]])
    for bar, val in zip(bars, df["ttft_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}s",
                ha="center", va="bottom")
    ax.set_ylabel("Cold-start TTFT (s)")
    ax.set_title("Cold-Start First-Request TTFT: Modal vs Simplismart")
    out = CHARTS_DIR / "coldstart_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_input_sweep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = df.groupby(["platform", "target_input_tokens"])["ttft_s"].mean().reset_index()
    for platform, grp in grouped.groupby("platform"):
        grp = grp.sort_values("target_input_tokens")
        ax.plot(grp["target_input_tokens"], grp["ttft_s"], marker="o",
                label=platform, color=COLORS.get(platform))
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Prefill Scaling: TTFT vs Input Length")
    ax.legend()
    out = CHARTS_DIR / "input_sweep_ttft.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_output_sweep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = df.groupby(["platform", "max_tokens"])["tpot_s"].mean().reset_index()
    for platform, grp in grouped.groupby("platform"):
        grp = grp.sort_values("max_tokens")
        ax.plot(grp["max_tokens"], grp["tpot_s"], marker="o",
                label=platform, color=COLORS.get(platform))
    ax.set_xlabel("Output length (max tokens)")
    ax.set_ylabel("TPOT (s/token)")
    ax.set_title("Decode Scaling: TPOT vs Output Length")
    ax.legend()
    out = CHARTS_DIR / "output_sweep_tpot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_cost_per_mtoken(conc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for platform, grp in conc.groupby("platform"):
        grp = grp.sort_values("concurrency")
        cost_per_mtok = (HOURLY_RATE[platform] / 3600) / grp["throughput_tok_s"] * 1_000_000
        ax.plot(grp["concurrency"], cost_per_mtok, marker="o",
                label=f"{platform} (${HOURLY_RATE[platform]:.2f}/hr)",
                color=COLORS.get(platform))
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Cost per 1M output tokens ($)")
    ax.set_title("Cost per Million Tokens vs Concurrency")
    ax.legend()
    out = CHARTS_DIR / "cost_per_mtoken_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(lat: pd.DataFrame, conc: pd.DataFrame, cold: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KEY NUMBERS SUMMARY")
    print("=" * 60)

    print("\n--- Warm single-stream latency (TTFT / TPOT) ---")
    for platform, grp in lat.groupby("platform"):
        ttft = grp["ttft_s"]
        tpot = grp["tpot_s"]
        print(f"{platform:12s} TTFT p50={ttft.quantile(.5):.3f}s "
              f"p90={ttft.quantile(.9):.3f}s p99={ttft.quantile(.99):.3f}s "
              f"(n={len(grp)})")
        print(f"{'':12s} TPOT p50={tpot.quantile(.5):.4f}s "
              f"p90={tpot.quantile(.9):.4f}s p99={tpot.quantile(.99):.4f}s")
        outliers = grp[grp["ttft_s"] > 10 * ttft.quantile(.5)]
        if not outliers.empty:
            clean_ttft = grp.loc[~grp.index.isin(outliers.index), "ttft_s"]
            print(f"{'':12s} NOTE: run(s) {list(outliers['run'])} show TTFT "
                  f">10x the median ({list(outliers['ttft_s'].round(1))}s) — "
                  f"likely a cold-start artifact, not warm latency. "
                  f"Excluding them: TTFT p50={clean_ttft.quantile(.5):.3f}s, "
                  f"max={clean_ttft.max():.3f}s (n={len(clean_ttft)})")

    print("\n--- Peak throughput (highest concurrency tested) ---")
    for platform, grp in conc.groupby("platform"):
        peak = grp.loc[grp["throughput_tok_s"].idxmax()]
        print(f"{platform:12s} peak={peak['throughput_tok_s']:.1f} tok/s "
              f"at concurrency={int(peak['concurrency'])}")

    print("\n--- Cold start (first-request TTFT after scale-to-zero) ---")
    for _, row in cold.iterrows():
        print(f"{row['platform']:12s} cold_wall={row['cold_wall_s']:.1f}s  "
              f"ttft={row['ttft_s']:.1f}s")

    print("\n--- Cost per 1M output tokens at peak concurrency ---")
    for platform, grp in conc.groupby("platform"):
        peak = grp.loc[grp["throughput_tok_s"].idxmax()]
        cost = (HOURLY_RATE[platform] / 3600) / peak["throughput_tok_s"] * 1_000_000
        print(f"{platform:12s} ${cost:.4f} / 1M tokens "
              f"(concurrency={int(peak['concurrency'])}, rate=${HOURLY_RATE[platform]:.2f}/hr)")
    print()


def main():
    lat = load_latency()
    conc = load_concurrency()
    cold = load_coldstart()
    in_sweep = load_input_sweep()
    out_sweep = load_output_sweep()

    if not lat.empty:
        plot_latency(lat)
    if not conc.empty:
        plot_throughput(conc)
        plot_cost_per_mtoken(conc)
    if not cold.empty:
        plot_coldstart(cold)
    if not in_sweep.empty:
        plot_input_sweep(in_sweep)
    if not out_sweep.empty:
        plot_output_sweep(out_sweep)

    print_summary(lat, conc, cold)


if __name__ == "__main__":
    main()
