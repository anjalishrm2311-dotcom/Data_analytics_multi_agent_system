import hmac
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt

from agents.cleaning_agent import cleaning_agent
from agents.orchestrator_agent import run_pipeline
from agents.ingestion_agent import load_bytes, fetch_api
from agents.chat_agent import ask_data
from agents.insights_agent import classify_columns, render_insights
from agents.reports_agent import render_report
from agents.prediction_agent import render_prediction

st.set_page_config(page_title="AI Autonomous Data Analytics", page_icon="🤖", layout="wide")

ss = st.session_state

# Deployment settings (set them as environment variables / Streamlit secrets):
#   PUBLIC_MODE=1   -> hides "Fetch from an API" and turns off code execution in the chat
#   APP_PASSWORD=.. -> asks for a password before the app opens
PUBLIC = os.environ.get("PUBLIC_MODE", "0") == "1"
PUBLIC_MAX_ROWS = 50_000  # free servers have limited memory, so big files are sampled in public mode


def check_password() -> bool:
    required = os.environ.get("APP_PASSWORD", "")
    if not required or ss.get("authed"):
        return True
    st.title("🔒 AI Autonomous Data Analytics")
    typed = st.text_input("Password", type="password")
    if typed:
        if hmac.compare_digest(typed.encode(), required.encode()):
            ss.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

PAGES = {
    "Dashboard": "🏠",
    "Data Analysis": "📊",
    "AI Chat": "💬",
    "Insights": "💡",
    "Prediction": "🤖",
    "Reports": "📄",
}


def set_data(df, source):
    """Save loaded data so it survives page switches."""
    ss.raw_df, ss.source = df, source
    ss.pop("cleaned_df", None)
    ss.pop("clean_report", None)
    ss.pop("result", None)
    ss.pop("calc_cache", None)
    ss.pop("load_note", None)
    ss.chat_messages = []


def cached(name, df, fn):
    """Run an expensive calculation once per dataset instead of on every click."""
    store = ss.setdefault("calc_cache", {})
    key = (name, id(df), df.shape)
    if key not in store:
        store[key] = fn()
    return store[key]


def count_duplicates(df) -> int:
    try:
        return int(df.duplicated().sum())
    except Exception:  # e.g. cells that hold lists
        return int(df.astype(str).duplicated().sum())


def short(n) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


def _fmt_value(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, dict):
        return ", ".join(f"{a}: {b}" for a, b in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    return str(v)


def render_clean_report(report):
    """Show the cleaning report as tables instead of raw JSON."""
    if not isinstance(report, dict):
        st.write(report)
        return
    filled, other = {}, {}
    for k, v in report.items():
        if k.endswith("_missing_filled") and isinstance(v, (int, float)):
            filled[k[: -len("_missing_filled")]] = int(v)
        else:
            other[k] = v
    if other:
        st.dataframe(
            pd.DataFrame([(k.replace("_", " ").capitalize(), _fmt_value(v)) for k, v in other.items()],
                         columns=["Step", "Result"]),
            width="stretch", hide_index=True,
        )
    if filled:
        st.markdown(f"**Missing values filled: {sum(filled.values()):,} cells in {len(filled)} columns**")
        st.dataframe(
            pd.DataFrame({"Column": list(filled), "Cells filled": list(filled.values())})
            .sort_values("Cells filled", ascending=False),
            width="stretch", hide_index=True,
        )


def require_data() -> bool:
    if "raw_df" not in ss:
        st.warning("Load data from the Dashboard first.")
        return False
    return True


def render_chat(df):
    """Chat UI: answers questions about the loaded dataset using Gemini (streamed)."""
    if "chat_messages" not in ss:
        ss.chat_messages = []

    if st.button("🗑️ Clear chat"):
        ss.chat_messages = []
        st.rerun()

    def show(m):
        with st.chat_message(m["role"]):
            if m.get("content"):
                st.markdown(m["content"])
            if m.get("table") is not None:
                st.dataframe(m["table"], width="stretch")
            if m.get("fig") is not None:
                st.pyplot(m["fig"])
            if m.get("code"):
                with st.expander("Show the code that was run"):
                    st.code(m["code"], language="python")

    for m in ss.chat_messages:
        show(m)

    question = st.chat_input("Ask anything about your dataset...")
    if question:
        history = list(ss.chat_messages)
        user_msg = {"role": "user", "content": question}
        ss.chat_messages.append(user_msg)
        show(user_msg)

        with st.spinner("Analysing..."):
            out = ask_data(df, question, history, raw_df=ss.get("raw_df"))
        reply = {
            "role": "assistant",
            "content": out["text"],
            "table": out["table"],
            "fig": out["fig"],
            "code": out["code"],
        }
        ss.chat_messages.append(reply)
        show(reply)


# ---------------- charts for the Data Analysis page ----------------
def _plot_distribution(df, col, categorical):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if categorical:
        df[col].value_counts().head(15).plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    else:
        df[col].dropna().plot(kind="hist", bins=30, ax=ax)
        ax.set_xlabel(str(col))
    ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    return fig


def _plot_corr(df, cols):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([str(c) for c in cols], fontsize=8)
    if len(cols) <= 10:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    return fig


def _plot_missing(df):
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    miss.plot(kind="bar", ax=ax)
    ax.set_ylabel("% missing")
    ax.set_title("Missing values by column")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🤖 AI Autonomous\n### Data Analytics Multi-Agent System")
    st.divider()
    page = st.radio("Navigation", list(PAGES), format_func=lambda p: f"{PAGES[p]}  {p}")
    st.divider()
    st.success("System Online")
    if PUBLIC:
        st.caption("Demo mode: please do not upload private or sensitive data.")
    if "source" in ss:
        st.caption(f"📂 Loaded: {ss.source}")
        if "raw_df" in ss:
            st.caption(f"{len(ss.raw_df):,} rows × {ss.raw_df.shape[1]} columns")

st.title("🤖 AI Autonomous Data Analytics")
st.caption("Intelligent • Autonomous • Multi-Agent • Data Analytics")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("Load your data")
    with st.container(border=True):
        mode = "Upload a file" if PUBLIC else st.radio(
            "Data source", ["Upload a file", "Fetch from an API"], horizontal=True
        )

        if mode == "Upload a file":
            f = st.file_uploader("Choose any file (CSV, Excel, JSON, XML, Parquet, PDF, TXT, ZIP...)")
            if f is not None and ss.get("source") != f.name:
                try:
                    loaded = load_bytes(f.getvalue(), f.name)
                    note = None
                    if PUBLIC and len(loaded) > PUBLIC_MAX_ROWS:
                        note = (
                            f"Demo mode: this file has {len(loaded):,} rows, so a random sample of "
                            f"{PUBLIC_MAX_ROWS:,} rows is used to stay within the free server's memory."
                        )
                        loaded = loaded.sample(PUBLIC_MAX_ROWS, random_state=0).reset_index(drop=True)
                    set_data(loaded, f.name)
                    ss.load_note = note
                except Exception as e:
                    st.error(f"Could not read this file: {e}")

        else:
            url = st.text_input("API URL", placeholder="https://api.example.com/data")
            c1, c2 = st.columns(2)
            auth = c1.selectbox("Authentication", ["None", "Bearer token", "Custom header", "Query parameter"])
            method = c2.selectbox("Method", ["GET", "POST"])

            key, key_name = "", ""
            if auth != "None":
                key = st.text_input("API key / token", type="password")
            if auth in ("Custom header", "Query parameter"):
                key_name = st.text_input("Key name", value="X-API-Key" if auth == "Custom header" else "api_key")

            extra = st.text_area("Extra headers (optional, JSON)", placeholder='{"Accept": "application/json"}')

            if st.button("🌐 Fetch data") and url:
                try:
                    headers = json.loads(extra) if extra.strip() else {}
                    with st.spinner("Calling API..."):
                        df, _ = fetch_api(url, method, auth, key, key_name, headers)
                    set_data(df, url)
                except Exception as e:
                    st.error(f"API request failed: {e}")

    if "raw_df" in ss:
        df = ss.raw_df
        st.success(f"✓ Data loaded from {ss.source}")
        if ss.get("load_note"):
            st.info(ss.load_note)

        if df.shape[1] == 1:
            st.warning(
                "Only 1 column was found. This looks like text, not a table. "
                "Insights, Prediction and Reports need a table with several columns."
            )

        n_missing = cached("missing", df, lambda: int(df.isnull().sum().sum()))
        n_dups = cached("dups", df, lambda: count_duplicates(df))
        metrics = [
            ("Rows", short(len(df)), f"{len(df):,}"),
            ("Columns", df.shape[1], None),
            ("Missing cells", short(n_missing), f"{n_missing:,}"),
            ("Duplicate rows", short(n_dups), f"{n_dups:,}"),
        ]
        for col, (label, value, full) in zip(st.columns(4), metrics):
            with col:
                with st.container(border=True):
                    st.metric(label, value, help=full)

        b1, b2 = st.columns(2)
        if b1.button("🧹 Clean data", width="stretch"):
            ss.cleaned_df, ss.clean_report = cleaning_agent(df.copy())

        if b2.button("🧠 Run orchestrator (full pipeline)", width="stretch"):
            with st.spinner("Agents working..."):
                ss.result = run_pipeline(df.copy())
                ss.cleaned_df = ss.result["cleaned_df"]

        tab_prev, tab_cols, tab_clean = st.tabs(["Preview", "Columns", "Cleaned data"])

        with tab_prev:
            st.dataframe(df.head(10), width="stretch")

        with tab_cols:
            def _info():
                return pd.DataFrame({
                    "Type": df.dtypes.astype(str),
                    "Missing": df.isna().sum(),
                    "Missing %": (df.isna().mean() * 100).round(1),
                    "Unique": df.nunique(),
                })
            st.dataframe(cached("info", df, _info), width="stretch")

        with tab_clean:
            if "cleaned_df" not in ss:
                st.info("Click **Clean data** or **Run orchestrator** to see the cleaned data here.")
            else:
                clean = ss.cleaned_df
                n_clean_missing = cached("missing", clean, lambda: int(clean.isna().sum().sum()))
                cm = [
                    ("Rows", f"{short(len(df))} → {short(len(clean))}"),
                    ("Columns", f"{df.shape[1]} → {clean.shape[1]}"),
                    ("Missing cells", f"{short(n_missing)} → {short(n_clean_missing)}"),
                ]
                for col, (label, value) in zip(st.columns(3), cm):
                    with col:
                        with st.container(border=True):
                            st.metric(label, value)
                st.dataframe(clean.head(10), width="stretch")
                if "clean_report" in ss:
                    with st.expander("Cleaning report"):
                        render_clean_report(ss.clean_report)
                if "result" in ss:
                    with st.expander("Agent decision trace", expanded=True):
                        for line in ss.result["decision_log"]:
                            st.write(line)
    else:
        st.info("👆 Upload a file or fetch from an API to start.")

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    if require_data():
        df = ss.get("cleaned_df", ss.raw_df)
        g = classify_columns(df)
        t_sum, t_dist, t_corr, t_miss = st.tabs(["Summary", "Distributions", "Correlation", "Missing values"])

        with t_sum:
            try:
                st.dataframe(cached("describe", df, lambda: df.describe(include="all").T), width="stretch")
            except Exception as e:
                st.info(f"Could not build the summary table: {e}")

        with t_dist:
            options = [c for c in df.columns if c not in g["id_like"] + g["text"] + g["constant"] + g["datetime"]]
            if not options:
                st.info("No columns suitable for a distribution chart.")
            else:
                col = st.selectbox("Column", options)
                fig = _plot_distribution(df, col, col in g["categorical"])
                st.pyplot(fig)
                plt.close(fig)

        with t_corr:
            if len(g["numeric"]) >= 2:
                cols = g["numeric"][:12]
                fig = _plot_corr(df, cols)
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(df[cols].corr().round(3), width="stretch")
            else:
                st.info("Correlation needs at least 2 continuous numeric columns.")

        with t_miss:
            if int(df.isna().sum().sum()) == 0:
                st.success("No missing values.")
            else:
                fig = _plot_missing(df)
                st.pyplot(fig)
                plt.close(fig)

# ---------------- AI CHAT ----------------
elif page == "AI Chat":
    st.header("💬 AI Data Analyst")
    if require_data():
        render_chat(ss.get("cleaned_df", ss.raw_df))

# ---------------- INSIGHTS ----------------
elif page == "Insights":
    st.header("💡 AI Insights")
    if require_data():
        render_insights(ss.get("cleaned_df", ss.raw_df))

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.header("🤖 Prediction")
    if require_data():
        render_prediction(ss.get("cleaned_df", ss.raw_df))

# ---------------- REPORTS ----------------
else:
    st.header("📄 Reports")
    if require_data():
        render_report(ss.get("cleaned_df", ss.raw_df), ss.raw_df)