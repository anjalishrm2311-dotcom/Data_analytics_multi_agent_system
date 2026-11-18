"""Insights agent: profiles ANY dataset with pandas, then asks Gemini to explain the facts."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


# ---------- 1. Understand the columns (works for any dataset) ----------
def classify_columns(df: pd.DataFrame) -> dict:
    n = len(df)
    g = {k: [] for k in ["numeric", "categorical", "datetime", "text", "id_like", "constant"]}
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            g["constant"].append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            g["datetime"].append(c)
        elif pd.api.types.is_bool_dtype(s):
            g["categorical"].append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if nun == n and pd.api.types.is_integer_dtype(s):
                g["id_like"].append(c)
            elif nun <= 10:
                g["categorical"].append(c)  # e.g. class, rating, count-like codes
            else:
                g["numeric"].append(c)
        elif nun == n and n > 20:
            g["id_like"].append(c)  # unique text per row (names, ids)
        elif nun <= 20:
            g["categorical"].append(c)
        else:
            g["text"].append(c)  # high-cardinality text
    return g


def data_signature(df: pd.DataFrame, target=None) -> str:
    return f"{df.shape}-{list(df.columns)}-{target}"


def _d(series: pd.Series, nd: int = 1) -> dict:
    return {str(k): (round(float(v), nd) if pd.notna(v) else None) for k, v in series.items()}


# ---------- 2. Facts (pure pandas, no AI) ----------
def _target_facts(df, target, num, cat) -> dict:
    s = df[target]
    out = {"target": str(target)}
    try:
        if s.nunique() == 2:
            pos = sorted(s.dropna().unique())[-1]
            y = (s == pos).astype(float).where(s.notna())
            out["positive_class"] = str(pos)
            out["positive_rate_pct"] = round(float(y.mean() * 100), 1)
            for c in [c for c in cat if c != target][:5]:
                out[f"positive_rate_by_{c}_pct"] = _d(y.groupby(df[c]).mean().mul(100))
            for c in num[:5]:
                out[f"average_{c}_by_target"] = _d(df.groupby(s)[c].mean(), 2)
        elif target in num:
            corr = df[num].corr()[target].drop(target, errors="ignore").dropna()
            top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(5)
            out["top_correlations_with_target"] = _d(top, 2)
            for c in [c for c in cat if c != target][:3]:
                out[f"average_target_by_{c}"] = _d(df.groupby(c)[target].mean(), 2)
        else:
            out["class_distribution_pct"] = _d(s.value_counts(normalize=True).head(8).mul(100))
    except Exception as e:  # keep the report alive even if one analysis fails
        out["note"] = f"target analysis partial: {e}"
    return out


def build_facts(df: pd.DataFrame, target=None) -> dict:
    g = classify_columns(df)
    num, cat = g["numeric"], g["categorical"]
    facts = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_groups": {k: [str(c) for c in v] for k, v in g.items() if v},
    }

    miss = df.isna().mean().mul(100).round(1)
    facts["missing_pct"] = {str(c): float(v) for c, v in miss.items() if v > 0}
    facts["duplicate_rows"] = int(df.duplicated().sum())

    if num:
        facts["numeric_summary"] = df[num[:12]].describe().round(2).to_dict()
        skew = df[num].skew().round(2)
        facts["highly_skewed_columns"] = {str(c): float(v) for c, v in skew.items() if abs(v) > 1}
        outs = {}
        for c in num:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            k = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
            if iqr > 0 and k > 0:
                outs[str(c)] = k
        facts["iqr_outlier_counts"] = outs

    if len(num) > 1:
        corr = df[num].corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pairs = corr.where(mask).stack()
        top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5)
        facts["top_correlations"] = {f"{a} vs {b}": round(float(v), 2) for (a, b), v in top.items()}

    if cat:
        facts["categorical_top_values_pct"] = {
            str(c): _d(df[c].value_counts(normalize=True).head(3).mul(100)) for c in cat[:8]
        }

    if g["datetime"]:
        facts["datetime_ranges"] = {str(c): [str(df[c].min()), str(df[c].max())] for c in g["datetime"]}

    if target is not None and target in df.columns:
        facts["target_analysis"] = _target_facts(df, target, num, cat)

    return facts


# ---------- 3. Charts (automatic for any dataset) ----------
def make_charts(df: pd.DataFrame, target=None, max_charts: int = 10):
    """Return a list of (caption, matplotlib figure)."""
    import matplotlib.pyplot as plt

    g = classify_columns(df)
    num, cat = g["numeric"], g["categorical"]
    charts = []

    def new():
        return plt.subplots(figsize=(5.5, 3.2))

    def add(caption, fig):
        fig.axes[0].set_title(caption, fontsize=10)
        fig.tight_layout()
        charts.append((caption, fig))

    def room():
        return len(charts) < max_charts

    # target-focused charts first
    if target is not None and target in df.columns:
        s = df[target]
        if s.nunique() == 2:
            pos = sorted(s.dropna().unique())[-1]
            y = (s == pos).astype(float).where(s.notna())
            for c in [c for c in cat if c != target][:3]:
                if not room():
                    break
                fig, ax = new()
                (y.groupby(df[c]).mean() * 100).plot(kind="bar", ax=ax)
                ax.set_ylabel(f"% {target} = {pos}")
                ax.tick_params(axis="x", rotation=30)
                add(f"{target} rate by {c}", fig)
        elif target in num:
            corr = df[num].corr()[target].drop(target, errors="ignore").dropna()
            if not corr.empty and room():
                fig, ax = new()
                corr.sort_values().plot(kind="barh", ax=ax)
                add(f"Correlation with {target}", fig)
        elif room():
            fig, ax = new()
            s.value_counts().head(10).plot(kind="bar", ax=ax)
            ax.tick_params(axis="x", rotation=30)
            add(f"Distribution of {target}", fig)

    # missing values
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False).head(15)
    if not miss.empty and room():
        fig, ax = new()
        miss.plot(kind="bar", ax=ax)
        ax.set_ylabel("% missing")
        ax.tick_params(axis="x", rotation=45)
        add("Missing values by column", fig)

    # numeric distributions
    for c in num[:4]:
        if not room():
            break
        fig, ax = new()
        df[c].dropna().plot(kind="hist", bins=30, ax=ax)
        ax.set_xlabel(str(c))
        add(f"Distribution of {c}", fig)

    # categorical counts
    for c in [c for c in cat if c != target][:4]:
        if not room():
            break
        fig, ax = new()
        df[c].value_counts().head(10).plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        add(f"Counts of {c}", fig)

    # correlation heatmap
    if len(num) >= 3 and room():
        cols = num[:10]
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels([str(c) for c in cols], fontsize=7)
        fig.colorbar(im, ax=ax)
        add("Correlation heatmap", fig)

    return charts


# ---------- 4. Gemini (only place that touches the SDK) ----------
def _call_gemini(prompt: str) -> str:
    from google import genai

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    last_error = None
    for model in DEFAULT_MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text
        except Exception as e:
            last_error = e
    raise last_error


def generate_insights(facts: dict) -> str:
    prompt = (
        "You are a senior data analyst. Below are facts computed with pandas for an uploaded dataset. "
        "Use ONLY these facts. You may guess the likely domain from the column names, but never assume "
        "anything the facts do not show. Never invent a number. If a fact group is empty, simply skip it; "
        "do not mention that it is empty. If the dataset has very few usable columns (for example a single "
        "text column), say so plainly and keep the analysis short.\n\n"
        "Write markdown with exactly three sections, using exactly these heading lines: "
        "'### Key findings', '### Data quality issues', '### Recommendations'. "
        "Under Key findings write 6-8 bullets that quote exact numbers. "
        "Under Data quality issues write bullets (or 'None found'). "
        "Under Recommendations write 4-5 concrete next steps. "
        "The heading lines must contain only the heading name, nothing else.\n\n"
        f"FACTS (JSON):\n{json.dumps(facts, indent=2, default=str)}"
    )
    return _call_gemini(prompt)


def generate_executive_summary(facts: dict) -> str:
    prompt = (
        "You are a senior data analyst. Using ONLY the facts below, write an executive summary of 3-4 "
        "plain-language sentences for a non-technical reader: what the dataset is about (guess from column "
        "names, carefully), its size and quality, and the single most important finding. "
        "Do not invent numbers. No bullets, no headings.\n\n"
        f"FACTS (JSON):\n{json.dumps(facts, indent=2, default=str)}"
    )
    return _call_gemini(prompt).strip()


# ---------- 5. Streamlit UI ----------
def target_picker(df: pd.DataFrame):
    """Optional 'what do you want to predict / explain' column, shared by Insights and Reports."""
    g = classify_columns(df)
    skip = set(g["id_like"] + g["text"] + g["constant"])  # not sensible as a target
    opts = ["(none)"] + [c for c in df.columns if c not in skip]
    cur = st.session_state.get("target_col", "(none)")
    idx = opts.index(cur) if cur in opts else 0
    choice = st.selectbox(
        "Target / outcome column (optional)",
        opts,
        index=idx,
        help="Pick the column you care about most (e.g. Survived, Price, Churn) to get focused analysis.",
    )
    st.session_state["target_col"] = choice
    return None if choice == "(none)" else choice


def _fmt(v):
    if isinstance(v, dict):
        return ", ".join(f"{a}: {b}" for a, b in v.items())
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


def render_facts(facts: dict):
    """Show the computed facts as readable tables (instead of raw JSON)."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{facts['rows']:,}")
    c2.metric("Columns", facts["columns"])
    c3.metric("Duplicate rows", f"{facts['duplicate_rows']:,}")

    groups = facts.get("column_groups", {})
    if groups:
        st.markdown("**Column types**")
        st.dataframe(
            pd.DataFrame([
                {"Type": k.replace("_", " "), "Count": len(v),
                 "Columns": ", ".join(v[:12]) + (" ..." if len(v) > 12 else "")}
                for k, v in groups.items()
            ]),
            width="stretch", hide_index=True,
        )
    if facts.get("missing_pct"):
        st.markdown("**Missing values (%)**")
        st.dataframe(pd.Series(facts["missing_pct"], name="Missing %").sort_values(ascending=False).head(30).to_frame(),
                     width="stretch")
    for key, title in [("top_correlations", "Strongest correlations"),
                       ("iqr_outlier_counts", "Possible outliers (IQR rule, number of rows)"),
                       ("highly_skewed_columns", "Highly skewed columns (skewness)")]:
        if facts.get(key):
            st.markdown(f"**{title}**")
            st.dataframe(pd.Series(facts[key], name="Value").to_frame(), width="stretch")
    if facts.get("numeric_summary"):
        st.markdown("**Numeric summary**")
        st.dataframe(pd.DataFrame(facts["numeric_summary"]).T, width="stretch")
    if facts.get("categorical_top_values_pct"):
        st.markdown("**Most common values of category columns (% of rows)**")
        st.dataframe(
            pd.DataFrame([{"Column": c, "Top values": ", ".join(f"{k} ({v}%)" for k, v in d.items())}
                          for c, d in facts["categorical_top_values_pct"].items()]),
            width="stretch", hide_index=True,
        )
    if facts.get("datetime_ranges"):
        st.markdown("**Date ranges**")
        st.dataframe(pd.DataFrame([{"Column": c, "From": r[0], "To": r[1]} for c, r in facts["datetime_ranges"].items()]),
                     width="stretch", hide_index=True)
    if facts.get("target_analysis"):
        st.markdown("**Target analysis**")
        st.dataframe(
            pd.DataFrame([{"Measure": k.replace("_", " "), "Result": _fmt(v)}
                          for k, v in facts["target_analysis"].items() if k != "target"]),
            width="stretch", hide_index=True,
        )


def render_insights(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    target = target_picker(df)
    sig = data_signature(df, target)
    if st.session_state.get("insights_sig") != sig:
        st.session_state.pop("insights_text", None)
        st.session_state.pop("insights_facts", None)
        st.session_state["insights_sig"] = sig

    st.caption("Facts are computed with pandas; Gemini only explains them.")
    if st.button("Generate Insights"):
        facts = build_facts(df, target)
        st.session_state["insights_facts"] = facts
        with st.spinner("Gemini is analysing the data..."):
            try:
                st.session_state["insights_text"] = generate_insights(facts)
            except Exception as e:
                st.error(f"Gemini error: {e}")

    text = st.session_state.get("insights_text")
    if text:
        st.markdown(text)
        with st.expander("Facts used (verify the numbers here)"):
            render_facts(st.session_state["insights_facts"])
        st.subheader("Supporting charts")
        cols = st.columns(2)
        for i, (_, fig) in enumerate(make_charts(df, target)):
            cols[i % 2].pyplot(fig)
            plt.close(fig)
        st.download_button("Download insights (.md)", text, file_name="insights.md")