"""All registered context features for the Alpha Discovery Engine.

Each feature is:

1. Implemented as a :class:`~apex_lab.research.context.registry.ContextFeature` subclass.
2. Registered immediately after definition via :func:`register_context_feature`.

The engine imports this module to trigger registration; it never references
individual feature classes by name.

Features are grouped into:

- **Trend Context** — EMA200/50 position and slope.
- **Volatility Context** — ATR percentile, state, realised vol, Bollinger width.
- **Momentum Context** — RSI, MACD histogram, ADX, ROC.
- **VWAP Context** — VWAP side, distance, slope.
- **Market Structure** — higher/lower highs and lows, swing distance.
- **Opening Session** — gap %, opening-range position.
- **Time Context** — hour, day-of-week, month, quarter.
"""

from __future__ import annotations

import polars as pl

from apex_lab.research.context.registry import ContextFeature, register_context_feature
from apex_lab.research.factors._utils import rolling_percentile_rank

# ---------------------------------------------------------------------------
# Bucketing helper
# ---------------------------------------------------------------------------


def _bucket(series: pl.Series, breaks: list[float], labels: list[str]) -> pl.Series:
    """Map a continuous series to labeled buckets.

    Args:
        series: Float64 input series (nulls allowed).
        breaks: Sorted break points. *N* breaks yield *N+1* buckets.
        labels: Bucket labels. Must have ``len(breaks) + 1`` elements.

    Returns:
        Utf8 Series of the same length; null where input is null.
    """
    assert len(labels) == len(breaks) + 1, "Need len(breaks)+1 labels"
    df = series.to_frame("_v")
    chain = pl.when(pl.col("_v").is_null()).then(pl.lit(None, dtype=pl.Utf8))
    for i, brk in enumerate(breaks):
        chain = chain.when(pl.col("_v") < brk).then(pl.lit(labels[i]))
    return df.select(chain.otherwise(pl.lit(labels[-1])).alias("b")).to_series()


# ---------------------------------------------------------------------------
# Shared indicator helpers (idempotent)
# ---------------------------------------------------------------------------


def _add_dist_ema200(df: pl.DataFrame) -> pl.DataFrame:
    if "_dist_ema200_pct" in df.columns:
        return df
    return df.with_columns(
        ((pl.col("close") - pl.col("ema_200")) / (pl.col("ema_200") + 1e-9) * 100.0).alias(
            "_dist_ema200_pct"
        )
    )


def _add_dist_ema50(df: pl.DataFrame) -> pl.DataFrame:
    if "_dist_ema50_pct" in df.columns:
        return df
    return df.with_columns(
        ((pl.col("close") - pl.col("ema_50")) / (pl.col("ema_50") + 1e-9) * 100.0).alias(
            "_dist_ema50_pct"
        )
    )


def _add_ema50_slope(df: pl.DataFrame) -> pl.DataFrame:
    if "_ema50_slope_pct" in df.columns:
        return df
    return df.with_columns(
        (
            (pl.col("ema_50") - pl.col("ema_50").shift(5))
            / (pl.col("ema_50").shift(5) + 1e-9)
            * 100.0
        ).alias("_ema50_slope_pct")
    )


def _add_ema200_slope(df: pl.DataFrame) -> pl.DataFrame:
    if "_ema200_slope_pct" in df.columns:
        return df
    return df.with_columns(
        (
            (pl.col("ema_200") - pl.col("ema_200").shift(5))
            / (pl.col("ema_200").shift(5) + 1e-9)
            * 100.0
        ).alias("_ema200_slope_pct")
    )


def _add_atr_state(df: pl.DataFrame) -> pl.DataFrame:
    if "_atr_mean_20" in df.columns:
        return df
    return df.with_columns(
        pl.col("atr_14").rolling_mean(window_size=20).alias("_atr_mean_20")
    )


def _add_realized_vol(df: pl.DataFrame) -> pl.DataFrame:
    if "_realized_vol_20" in df.columns:
        return df
    close_ret = pl.col("close").pct_change() * 100.0
    return df.with_columns(
        close_ret.rolling_std(window_size=20).alias("_realized_vol_20")
    )


def _add_bb_width_pct(df: pl.DataFrame) -> pl.DataFrame:
    if "_bb_width_pct" in df.columns:
        return df
    df2 = df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=20).alias("_bb_mid"),
            pl.col("close").rolling_std(window_size=20).alias("_bb_std"),
        ]
    ).with_columns(
        [
            (pl.col("_bb_mid") + 2.0 * pl.col("_bb_std")).alias("_bb_upper"),
            (pl.col("_bb_mid") - 2.0 * pl.col("_bb_std")).alias("_bb_lower"),
        ]
    ).with_columns(
        (
            (pl.col("_bb_upper") - pl.col("_bb_lower")) / (pl.col("_bb_mid") + 1e-9)
        ).alias("_bb_width")
    )
    bb_width_pct = rolling_percentile_rank(df2.get_column("_bb_width"), window=100)
    return df2.with_columns(bb_width_pct.alias("_bb_width_pct"))


def _add_macd_hist_pct(df: pl.DataFrame) -> pl.DataFrame:
    if "_macd_hist_pct" in df.columns:
        return df
    macd_hist_pct = rolling_percentile_rank(df.get_column("macd_hist"), window=100)
    return df.with_columns(macd_hist_pct.alias("_macd_hist_pct"))


def _add_adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    if "adx_14" in df.columns:
        return df
    alpha = 1.0 / period
    df2 = df.with_columns(
        [
            pl.max_horizontal(
                [
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ]
            )
            .cast(pl.Float64)
            .alias("_adx_tr"),
            pl.when(
                (pl.col("high") - pl.col("high").shift(1) > pl.col("low").shift(1) - pl.col("low"))
                & (pl.col("high") - pl.col("high").shift(1) > 0)
            )
            .then(pl.col("high") - pl.col("high").shift(1))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64)
            .alias("_adx_dm_p"),
            pl.when(
                (pl.col("low").shift(1) - pl.col("low") > pl.col("high") - pl.col("high").shift(1))
                & (pl.col("low").shift(1) - pl.col("low") > 0)
            )
            .then(pl.col("low").shift(1) - pl.col("low"))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64)
            .alias("_adx_dm_m"),
        ]
    ).with_columns(
        [
            pl.col("_adx_tr").ewm_mean(alpha=alpha, adjust=False).alias("_adx_str"),
            pl.col("_adx_dm_p").ewm_mean(alpha=alpha, adjust=False).alias("_adx_sdm_p"),
            pl.col("_adx_dm_m").ewm_mean(alpha=alpha, adjust=False).alias("_adx_sdm_m"),
        ]
    ).with_columns(
        [
            (pl.col("_adx_sdm_p") / (pl.col("_adx_str") + 1e-9) * 100.0).alias("_adx_di_p"),
            (pl.col("_adx_sdm_m") / (pl.col("_adx_str") + 1e-9) * 100.0).alias("_adx_di_m"),
        ]
    ).with_columns(
        (
            (pl.col("_adx_di_p") - pl.col("_adx_di_m")).abs()
            / (pl.col("_adx_di_p") + pl.col("_adx_di_m") + 1e-9)
            * 100.0
        ).alias("_adx_dx")
    ).with_columns(
        pl.col("_adx_dx").ewm_mean(alpha=alpha, adjust=False).alias("adx_14")
    ).drop(
        ["_adx_tr", "_adx_dm_p", "_adx_dm_m", "_adx_str", "_adx_sdm_p", "_adx_sdm_m",
         "_adx_di_p", "_adx_di_m", "_adx_dx"]
    )
    return df2


def _add_roc(df: pl.DataFrame) -> pl.DataFrame:
    cols_needed = {"_roc_10", "_roc_20"}
    if cols_needed.issubset(df.columns):
        return df
    return df.with_columns(
        [
            (
                (pl.col("close") - pl.col("close").shift(10))
                / (pl.col("close").shift(10) + 1e-9)
                * 100.0
            ).alias("_roc_10"),
            (
                (pl.col("close") - pl.col("close").shift(20))
                / (pl.col("close").shift(20) + 1e-9)
                * 100.0
            ).alias("_roc_20"),
        ]
    )


def _add_dist_vwap(df: pl.DataFrame) -> pl.DataFrame:
    if "_dist_vwap_pct" in df.columns:
        return df
    return df.with_columns(
        ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 1e-9) * 100.0).alias(
            "_dist_vwap_pct"
        )
    )


def _add_vwap_slope(df: pl.DataFrame) -> pl.DataFrame:
    if "_vwap_slope_pct" in df.columns:
        return df
    return df.with_columns(
        (
            (pl.col("vwap") - pl.col("vwap").shift(3))
            / (pl.col("vwap").shift(3) + 1e-9)
            * 100.0
        ).alias("_vwap_slope_pct")
    )


def _add_swing_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling 10-bar swing high/low reference columns (idempotent)."""
    if "_swing_high_10" in df.columns:
        return df
    return df.with_columns(
        [
            pl.col("high").shift(1).rolling_max(window_size=10).alias("_swing_high_10"),
            pl.col("low").shift(1).rolling_min(window_size=10).alias("_swing_low_10"),
        ]
    )


def _add_swing_dist(df: pl.DataFrame) -> pl.DataFrame:
    """Add 20-bar swing range distance columns (idempotent)."""
    if "_swing_dist_pct" in df.columns:
        return df
    df2 = df.with_columns(
        [
            pl.col("high").rolling_max(window_size=20).alias("_swing_range_high"),
            pl.col("low").rolling_min(window_size=20).alias("_swing_range_low"),
        ]
    )
    return df2.with_columns(
        (
            (pl.col("close") - pl.col("_swing_range_low"))
            / (pl.col("_swing_range_high") - pl.col("_swing_range_low") + 1e-9)
            * 100.0
        ).alias("_swing_dist_pct")
    )


def _add_opening_session(df: pl.DataFrame) -> pl.DataFrame:
    """Add or_high, or_low, gap_pct columns per trading session (idempotent).

    Uses the FIRST bar of each calendar date as the opening range.
    Gap % = (first-bar open − previous day last close) / previous day last close × 100.
    All opening-session values are propagated to every bar of the same date.
    """
    if "or_high" in df.columns:
        return df

    df_date = df.with_columns(pl.col("timestamp").dt.date().alias("_date"))

    # Per-day opening range (first bar's high/low/open) and day's last close
    or_per_day = (
        df_date.sort("timestamp")
        .group_by("_date", maintain_order=True)
        .agg(
            [
                pl.col("high").first().alias("or_high"),
                pl.col("low").first().alias("or_low"),
                pl.col("open").first().alias("_or_open"),
                pl.col("close").last().alias("_day_close"),
            ]
        )
        .sort("_date")
    )

    # Shift day close to get previous-day close for gap computation
    or_per_day = or_per_day.with_columns(
        pl.col("_day_close").shift(1).alias("_prev_close")
    ).with_columns(
        (
            (pl.col("_or_open") - pl.col("_prev_close"))
            / (pl.col("_prev_close") + 1e-9)
            * 100.0
        ).alias("gap_pct")
    )

    result = (
        df_date.join(
            or_per_day.select(["_date", "or_high", "or_low", "gap_pct"]),
            on="_date",
            how="left",
        )
        .drop("_date")
    )
    return result


# ---------------------------------------------------------------------------
# ============================================================
# TREND CONTEXT
# ============================================================
# ---------------------------------------------------------------------------


class _EMA200Side(ContextFeature):
    @property
    def name(self) -> str:
        return "ema200_side"

    @property
    def group(self) -> str:
        return "trend"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # ema_200 already added by _base_enrich

    def label(self, df: pl.DataFrame) -> pl.Series:
        above = (df["close"] > df["ema_200"]).fill_null(False)
        return above.map_elements(
            lambda x: "Above EMA200" if x else "Below EMA200", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["close"] > df["ema_200"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_EMA200Side())


class _DistEMA200(ContextFeature):
    @property
    def name(self) -> str:
        return "dist_ema200"

    @property
    def group(self) -> str:
        return "trend"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_dist_ema200(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_dist_ema200_pct"],
            breaks=[-5.0, -2.0, 0.0, 2.0, 5.0],
            labels=["<-5%", "-5% to -2%", "-2% to 0%", "0% to 2%", "2% to 5%", ">5%"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_dist_ema200_pct"]


register_context_feature(_DistEMA200())


class _DistEMA50(ContextFeature):
    @property
    def name(self) -> str:
        return "dist_ema50"

    @property
    def group(self) -> str:
        return "trend"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_dist_ema50(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_dist_ema50_pct"],
            breaks=[-3.0, -1.0, 0.0, 1.0, 3.0],
            labels=["<-3%", "-3% to -1%", "-1% to 0%", "0% to 1%", "1% to 3%", ">3%"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_dist_ema50_pct"]


register_context_feature(_DistEMA50())


class _EMA50Slope(ContextFeature):
    @property
    def name(self) -> str:
        return "ema50_slope"

    @property
    def group(self) -> str:
        return "trend"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_ema50_slope(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_ema50_slope_pct"],
            breaks=[-0.1, 0.1],
            labels=["Falling (<-0.1%)", "Flat (-0.1% to 0.1%)", "Rising (>0.1%)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_ema50_slope_pct"]


register_context_feature(_EMA50Slope())


class _EMA200Slope(ContextFeature):
    @property
    def name(self) -> str:
        return "ema200_slope"

    @property
    def group(self) -> str:
        return "trend"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_ema200_slope(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_ema200_slope_pct"],
            breaks=[-0.05, 0.05],
            labels=["Falling (<-0.05%)", "Flat (-0.05% to 0.05%)", "Rising (>0.05%)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_ema200_slope_pct"]


register_context_feature(_EMA200Slope())


# ---------------------------------------------------------------------------
# ============================================================
# VOLATILITY CONTEXT
# ============================================================
# ---------------------------------------------------------------------------


class _ATRPercentile(ContextFeature):
    @property
    def name(self) -> str:
        return "atr_percentile"

    @property
    def group(self) -> str:
        return "volatility"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # atr_pct already added by _base_enrich

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["atr_pct"],
            breaks=[20.0, 40.0, 60.0, 80.0],
            labels=[
                "0-20 (Very Low)",
                "20-40 (Low)",
                "40-60 (Average)",
                "60-80 (High)",
                "80-100 (Very High)",
            ],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["atr_pct"]


register_context_feature(_ATRPercentile())


class _ATRState(ContextFeature):
    """ATR expansion / contraction regime vs its 20-bar mean."""

    @property
    def name(self) -> str:
        return "atr_state"

    @property
    def group(self) -> str:
        return "volatility"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_atr_state(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        ratio = df["atr_14"] / (df["_atr_mean_20"] + 1e-9)
        df2 = ratio.to_frame("_r")
        chain = (
            pl.when(pl.col("_r").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_r") > 1.1)
            .then(pl.lit("Expansion"))
            .when(pl.col("_r") < 0.9)
            .then(pl.lit("Contraction"))
            .otherwise(pl.lit("Neutral"))
        )
        return df2.select(chain.alias("s")).to_series()

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["atr_14"] / (df["_atr_mean_20"] + 1e-9)).cast(pl.Float64)


register_context_feature(_ATRState())


class _RealizedVol20(ContextFeature):
    @property
    def name(self) -> str:
        return "realized_vol_20"

    @property
    def group(self) -> str:
        return "volatility"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_realized_vol(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_realized_vol_20"],
            breaks=[0.5, 1.0, 2.0, 3.0],
            labels=["<0.5% (Very Low)", "0.5-1% (Low)", "1-2% (Medium)", "2-3% (High)", ">3% (Very High)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_realized_vol_20"]


register_context_feature(_RealizedVol20())


class _BBWidthPct(ContextFeature):
    @property
    def name(self) -> str:
        return "bb_width_pct"

    @property
    def group(self) -> str:
        return "volatility"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_bb_width_pct(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_bb_width_pct"],
            breaks=[20.0, 40.0, 60.0, 80.0],
            labels=[
                "0-20 (Very Narrow)",
                "20-40 (Narrow)",
                "40-60 (Average)",
                "60-80 (Wide)",
                "80-100 (Very Wide)",
            ],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_bb_width_pct"]


register_context_feature(_BBWidthPct())


# ---------------------------------------------------------------------------
# ============================================================
# MOMENTUM CONTEXT
# ============================================================
# ---------------------------------------------------------------------------


class _RSIBucket(ContextFeature):
    @property
    def name(self) -> str:
        return "rsi_bucket"

    @property
    def group(self) -> str:
        return "momentum"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # rsi_14 already added by _base_enrich

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["rsi_14"],
            breaks=[30.0, 40.0, 50.0, 60.0, 70.0],
            labels=["0-30 (Oversold)", "30-40", "40-50", "50-60", "60-70", "70-100 (Overbought)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["rsi_14"]


register_context_feature(_RSIBucket())


class _MACDHistBucket(ContextFeature):
    """MACD histogram bucketed by its rolling 100-bar percentile rank."""

    @property
    def name(self) -> str:
        return "macd_hist_bucket"

    @property
    def group(self) -> str:
        return "momentum"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_macd_hist_pct(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_macd_hist_pct"],
            breaks=[20.0, 40.0, 60.0, 80.0],
            labels=[
                "0-20 (Strong Negative)",
                "20-40 (Weak Negative)",
                "40-60 (Near Zero)",
                "60-80 (Weak Positive)",
                "80-100 (Strong Positive)",
            ],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_macd_hist_pct"]


register_context_feature(_MACDHistBucket())


class _ADXBucket(ContextFeature):
    @property
    def name(self) -> str:
        return "adx_bucket"

    @property
    def group(self) -> str:
        return "momentum"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_adx(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["adx_14"],
            breaks=[15.0, 25.0, 35.0],
            labels=["0-15 (Weak)", "15-25 (Developing)", "25-35 (Strong)", "35+ (Very Strong)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["adx_14"]


register_context_feature(_ADXBucket())


class _ROC10Bucket(ContextFeature):
    @property
    def name(self) -> str:
        return "roc10_bucket"

    @property
    def group(self) -> str:
        return "momentum"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_roc(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_roc_10"],
            breaks=[-3.0, -1.0, 0.0, 1.0, 3.0],
            labels=["<-3%", "-3% to -1%", "-1% to 0%", "0% to 1%", "1% to 3%", ">3%"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_roc_10"]


register_context_feature(_ROC10Bucket())


class _ROC20Bucket(ContextFeature):
    @property
    def name(self) -> str:
        return "roc20_bucket"

    @property
    def group(self) -> str:
        return "momentum"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_roc(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_roc_20"],
            breaks=[-5.0, -2.0, 0.0, 2.0, 5.0],
            labels=["<-5%", "-5% to -2%", "-2% to 0%", "0% to 2%", "2% to 5%", ">5%"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_roc_20"]


register_context_feature(_ROC20Bucket())


# ---------------------------------------------------------------------------
# ============================================================
# VWAP CONTEXT
# ============================================================
# ---------------------------------------------------------------------------


class _VWAPSide(ContextFeature):
    @property
    def name(self) -> str:
        return "vwap_side"

    @property
    def group(self) -> str:
        return "vwap"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # vwap added by _base_enrich

    def label(self, df: pl.DataFrame) -> pl.Series:
        above = (df["close"] > df["vwap"]).fill_null(False)
        return above.map_elements(
            lambda x: "Above VWAP" if x else "Below VWAP", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["close"] > df["vwap"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_VWAPSide())


class _DistVWAP(ContextFeature):
    @property
    def name(self) -> str:
        return "dist_vwap"

    @property
    def group(self) -> str:
        return "vwap"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_dist_vwap(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_dist_vwap_pct"],
            breaks=[-2.0, -0.5, 0.0, 0.5, 2.0],
            labels=["<-2%", "-2% to -0.5%", "-0.5% to 0%", "0% to 0.5%", "0.5% to 2%", ">2%"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_dist_vwap_pct"]


register_context_feature(_DistVWAP())


class _VWAPSlope(ContextFeature):
    @property
    def name(self) -> str:
        return "vwap_slope"

    @property
    def group(self) -> str:
        return "vwap"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_vwap_slope(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_vwap_slope_pct"],
            breaks=[-0.05, 0.05],
            labels=["Falling (<-0.05%)", "Flat (-0.05% to 0.05%)", "Rising (>0.05%)"],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_vwap_slope_pct"]


register_context_feature(_VWAPSlope())


# ---------------------------------------------------------------------------
# ============================================================
# MARKET STRUCTURE
# ============================================================
# ---------------------------------------------------------------------------


class _HigherHigh(ContextFeature):
    @property
    def name(self) -> str:
        return "higher_high"

    @property
    def group(self) -> str:
        return "market_structure"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_swing_cols(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        hh = (df["high"] > df["_swing_high_10"]).fill_null(False)
        return hh.map_elements(
            lambda x: "Higher High" if x else "No Higher High", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["high"] > df["_swing_high_10"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_HigherHigh())


class _HigherLow(ContextFeature):
    @property
    def name(self) -> str:
        return "higher_low"

    @property
    def group(self) -> str:
        return "market_structure"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_swing_cols(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        hl = (df["low"] > df["_swing_low_10"]).fill_null(False)
        return hl.map_elements(
            lambda x: "Higher Low" if x else "No Higher Low", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["low"] > df["_swing_low_10"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_HigherLow())


class _LowerHigh(ContextFeature):
    @property
    def name(self) -> str:
        return "lower_high"

    @property
    def group(self) -> str:
        return "market_structure"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_swing_cols(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        lh = (df["high"] < df["_swing_high_10"]).fill_null(False)
        return lh.map_elements(
            lambda x: "Lower High" if x else "No Lower High", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["high"] < df["_swing_high_10"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_LowerHigh())


class _LowerLow(ContextFeature):
    @property
    def name(self) -> str:
        return "lower_low"

    @property
    def group(self) -> str:
        return "market_structure"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_swing_cols(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        ll = (df["low"] < df["_swing_low_10"]).fill_null(False)
        return ll.map_elements(
            lambda x: "Lower Low" if x else "No Lower Low", return_dtype=pl.Utf8
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return (df["low"] < df["_swing_low_10"]).cast(pl.Float64).fill_null(0.0)


register_context_feature(_LowerLow())


class _SwingDistance(ContextFeature):
    """Position of close within the 20-bar high–low range (0 = at range low, 100 = at range high)."""

    @property
    def name(self) -> str:
        return "swing_distance"

    @property
    def group(self) -> str:
        return "market_structure"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_swing_dist(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["_swing_dist_pct"],
            breaks=[20.0, 40.0, 60.0, 80.0],
            labels=[
                "0-20% (Near Low)",
                "20-40% (Low-Mid)",
                "40-60% (Mid)",
                "60-80% (High-Mid)",
                "80-100% (Near High)",
            ],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["_swing_dist_pct"]


register_context_feature(_SwingDistance())


# ---------------------------------------------------------------------------
# ============================================================
# OPENING SESSION
# ============================================================
# ---------------------------------------------------------------------------


class _GapPctBucket(ContextFeature):
    @property
    def name(self) -> str:
        return "gap_pct_bucket"

    @property
    def group(self) -> str:
        return "opening_session"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_opening_session(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        return _bucket(
            df["gap_pct"],
            breaks=[-1.0, -0.3, 0.3, 1.0],
            labels=[
                "Large Gap Down (<-1%)",
                "Small Gap Down (-1% to -0.3%)",
                "Flat (-0.3% to 0.3%)",
                "Small Gap Up (0.3% to 1%)",
                "Large Gap Up (>1%)",
            ],
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["gap_pct"]


register_context_feature(_GapPctBucket())


class _ORPosition(ContextFeature):
    """Position of close price relative to the day's opening range."""

    @property
    def name(self) -> str:
        return "or_position"

    @property
    def group(self) -> str:
        return "opening_session"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_opening_session(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        df2 = df.select(["close", "or_high", "or_low"])
        chain = (
            pl.when(pl.col("close").is_null() | pl.col("or_high").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("close") > pl.col("or_high"))
            .then(pl.lit("Above OR High"))
            .when(pl.col("close") < pl.col("or_low"))
            .then(pl.lit("Below OR Low"))
            .otherwise(pl.lit("Inside OR"))
        )
        return df2.select(chain.alias("s")).to_series()

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        # Normalised position: (close - or_low) / (or_high - or_low + eps) * 100
        return (
            (df["close"] - df["or_low"]) / (df["or_high"] - df["or_low"] + 1e-9) * 100.0
        ).cast(pl.Float64)


register_context_feature(_ORPosition())


class _InsideOR(ContextFeature):
    """Whether close is inside or outside the opening range."""

    @property
    def name(self) -> str:
        return "inside_or"

    @property
    def group(self) -> str:
        return "opening_session"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_opening_session(df)

    def label(self, df: pl.DataFrame) -> pl.Series:
        df2 = df.select(["close", "or_high", "or_low"])
        chain = (
            pl.when(pl.col("close").is_null() | pl.col("or_high").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when((pl.col("close") >= pl.col("or_low")) & (pl.col("close") <= pl.col("or_high")))
            .then(pl.lit("Inside Opening Range"))
            .otherwise(pl.lit("Outside Opening Range"))
        )
        return df2.select(chain.alias("s")).to_series()

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        inside = (df["close"] >= df["or_low"]) & (df["close"] <= df["or_high"])
        return inside.cast(pl.Float64).fill_null(0.0)


register_context_feature(_InsideOR())


# ---------------------------------------------------------------------------
# ============================================================
# TIME CONTEXT
# ============================================================
# ---------------------------------------------------------------------------

_DOW_NAMES = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday",
}
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


class _HourFeature(ContextFeature):
    @property
    def name(self) -> str:
        return "hour"

    @property
    def group(self) -> str:
        return "time"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def label(self, df: pl.DataFrame) -> pl.Series:
        return (
            df["timestamp"]
            .dt.hour()
            .map_elements(lambda x: f"{x:02d}", return_dtype=pl.Utf8)
            .alias("hour")
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["timestamp"].dt.hour().cast(pl.Float64)


register_context_feature(_HourFeature())


class _DayOfWeekFeature(ContextFeature):
    @property
    def name(self) -> str:
        return "day_of_week"

    @property
    def group(self) -> str:
        return "time"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def label(self, df: pl.DataFrame) -> pl.Series:
        return (
            df["timestamp"]
            .dt.weekday()
            .map_elements(lambda x: _DOW_NAMES.get(x, str(x)), return_dtype=pl.Utf8)
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["timestamp"].dt.weekday().cast(pl.Float64)


register_context_feature(_DayOfWeekFeature())


class _MonthFeature(ContextFeature):
    @property
    def name(self) -> str:
        return "month"

    @property
    def group(self) -> str:
        return "time"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def label(self, df: pl.DataFrame) -> pl.Series:
        return (
            df["timestamp"]
            .dt.month()
            .map_elements(lambda x: _MONTH_NAMES.get(x, str(x)), return_dtype=pl.Utf8)
        )

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return df["timestamp"].dt.month().cast(pl.Float64)


register_context_feature(_MonthFeature())


class _QuarterFeature(ContextFeature):
    @property
    def name(self) -> str:
        return "quarter"

    @property
    def group(self) -> str:
        return "time"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def label(self, df: pl.DataFrame) -> pl.Series:
        quarters = ((df["timestamp"].dt.month() - 1) // 3 + 1).map_elements(
            lambda x: f"Q{x}", return_dtype=pl.Utf8
        )
        return quarters

    def numeric(self, df: pl.DataFrame) -> pl.Series:
        return ((df["timestamp"].dt.month() - 1) // 3 + 1).cast(pl.Float64)


register_context_feature(_QuarterFeature())
