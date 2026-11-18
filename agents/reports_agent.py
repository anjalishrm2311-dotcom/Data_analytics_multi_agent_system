"""Reports agent: builds a professional Word (.docx) report for ANY uploaded dataset."""
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from agents.insights_agent import (
    build_facts,
    classify_columns,
    data_signature,
    generate_executive_summary,
    generate_insights,
    make_charts,
    target_picker,
)

KIND_LABEL = {
    "numeric": "Numeric",
    "categorical": "Categorical",
    "datetime": "Date/time",
    "text": "Text",
    "id_like": "ID-like",
    "constant": "Constant",
}


# ---------- small docx helpers ----------
# Characters that Word/XML cannot store (NULL bytes, control characters). Real-world data often has them.
_BAD_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _clean(x):
    return _BAD_XML.sub("", str(x))


def _p(doc, text="", **kw):
    return doc.add_paragraph(_clean(text), **kw)


def _add_page_number(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Page ")
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def _table(doc, headers, rows, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = _clean(h)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = _clean(v)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(font_size)
    _p(doc, )


def _markdown(doc, text):
    """Tiny markdown -> docx: ### headings, bullets, bold markers removed."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        if clean.startswith("#"):
            doc.add_heading(_clean(clean.lstrip("# ").strip()), 2)
        elif clean.startswith(("- ", "* ")):
            _p(doc, clean[2:], style="List Bullet")
        else:
            _p(doc, clean)


def _fmt(v):
    if isinstance(v, dict):
        return ", ".join(f"{a}: {b}" for a, b in v.items())
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


def _col_summary(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return f"mean {s.mean():.2f}, min {s.min():.4g}, max {s.max():.4g}"
    mode = s.mode(dropna=True)
    return f"most common: {str(mode.iat[0])[:30]}" if not mode.empty else "-"


def _cleaning_section(doc, raw, clean):
    if raw is clean:
        _p(doc, "No cleaning was applied: the report uses the original data.")
        return
    _table(
        doc,
        ["Metric", "Before cleaning", "After cleaning"],
        [
            ("Rows", len(raw), len(clean)),
            ("Columns", raw.shape[1], clean.shape[1]),
            ("Missing cells", int(raw.isna().sum().sum()), int(clean.isna().sum().sum())),
            ("Duplicate rows", int(raw.duplicated().sum()), int(clean.duplicated().sum())),
        ],
    )
    dropped = [c for c in raw.columns if c not in clean.columns]
    added = [c for c in clean.columns if c not in raw.columns]
    if dropped:
        _p(doc, "Columns removed: " + ", ".join(map(str, dropped)))
    if added:
        _p(doc, "Columns added: " + ", ".join(map(str, added)))
    common = [c for c in raw.columns if c in clean.columns]
    filled = [
        (c, int(raw[c].isna().sum()), int(clean[c].isna().sum()))
        for c in common
        if raw[c].isna().sum() != clean[c].isna().sum()
    ]
    if filled:
        _p(doc, "Missing values handled:")
        _table(doc, ["Column", "Missing before", "Missing after"], filled)
    retyped = [(c, str(raw[c].dtype), str(clean[c].dtype)) for c in common if raw[c].dtype != clean[c].dtype]
    if retyped:
        _p(doc, "Data types changed:")
        _table(doc, ["Column", "Before", "After"], retyped)
    if not (dropped or added or filled or retyped) and len(raw) == len(clean):
        _p(doc, "Cleaning made no visible changes to this dataset.")


# ---------- the report ----------
def build_report(df, raw_df, insights_text, exec_summary, title, subtitle, author, target, ml=None) -> bytes:
    facts = build_facts(df, target)
    g = classify_columns(df)
    kind_of = {c: KIND_LABEL[k] for k, cols in g.items() for c in cols}

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    _add_page_number(doc)

    # cover page
    for _ in range(6):
        _p(doc, )
    p = _p(doc, )
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(_clean(title))
    r.bold = True
    r.font.size = Pt(30)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    if subtitle:
        p = _p(doc, )
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(_clean(subtitle)).font.size = Pt(16)
    p = _p(doc, )
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = f"Generated on {datetime.now():%d %B %Y}"
    if author:
        meta += f"  |  Prepared by {author}"
    p.add_run(_clean(meta)).font.size = Pt(11)
    doc.add_page_break()

    n = [0]

    def h(text):
        n[0] += 1
        doc.add_heading(_clean(f"{n[0]}. {text}"), 1)

    # 1. Executive summary
    h("Executive summary")
    miss_total = int(df.isna().sum().sum())
    miss_pct = miss_total / max(df.size, 1) * 100
    _p(doc, 
        exec_summary
        or (
            f"The dataset has {len(df):,} rows and {df.shape[1]} columns, with {miss_total:,} missing cells "
            f"({miss_pct:.1f}%) and {facts['duplicate_rows']} duplicate rows."
        )
    )

    # 2. Dataset overview
    h("Dataset overview")
    rows = [
        ("Rows", f"{len(df):,}"),
        ("Columns", df.shape[1]),
        ("Numeric columns", len(g["numeric"])),
        ("Categorical columns", len(g["categorical"])),
        ("Date/time columns", len(g["datetime"])),
        ("Text columns", len(g["text"])),
        ("ID-like columns", len(g["id_like"])),
        ("Constant columns", len(g["constant"])),
        ("Missing cells", f"{miss_total:,} ({miss_pct:.1f}%)"),
        ("Duplicate rows", facts["duplicate_rows"]),
        ("Memory usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"),
    ]
    if target is not None:
        rows.append(("Target column", target))
    _table(doc, ["Item", "Value"], rows)

    # 3. Cleaning summary
    if raw_df is not None:
        h("Data cleaning summary")
        _cleaning_section(doc, raw_df, df)

    # 4. Data quality
    h("Data quality")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        _p(doc, "No missing values.")
    else:
        _table(
            doc,
            ["Column", "Missing", "Missing %"],
            [(c, int(v), f"{v / len(df) * 100:.1f}%") for c, v in miss.items()],
        )
    for label, cols in [
        ("Constant columns (carry no information)", g["constant"]),
        ("ID-like columns (unique per row, not useful for analysis)", g["id_like"]),
        ("High-cardinality text columns", g["text"]),
    ]:
        if cols:
            _p(doc, f"{label}: " + ", ".join(map(str, cols)))
    outs = facts.get("iqr_outlier_counts", {})
    if outs:
        _p(doc, "Possible outliers (IQR rule):")
        _table(doc, ["Column", "Outliers", "% of rows"], [(c, k, f"{k / len(df) * 100:.1f}%") for c, k in outs.items()])
    skew = facts.get("highly_skewed_columns", {})
    if skew:
        _p(doc, "Highly skewed columns (|skew| > 1): " + ", ".join(f"{c} ({v})" for c, v in skew.items()))

    # 5. Column profile
    h("Column profile")
    cols = list(df.columns[:40])
    _table(
        doc,
        ["Column", "Type", "Unique", "Missing %", "Summary"],
        [
            (c, kind_of.get(c, "-"), int(df[c].nunique()), f"{df[c].isna().mean() * 100:.1f}%", _col_summary(df[c]))
            for c in cols
        ],
        font_size=8,
    )
    if df.shape[1] > 40:
        _p(doc, f"Showing the first 40 of {df.shape[1]} columns.")

    # 6. Numeric summary
    h("Numeric summary")
    num = g["numeric"][:15]
    if num:
        desc = df[num].describe().round(2).T[["count", "mean", "std", "min", "50%", "max"]]
        _table(
            doc,
            ["Column", "Count", "Mean", "Std", "Min", "Median", "Max"],
            [(c, *[f"{v:g}" for v in r]) for c, r in desc.iterrows()],
        )
    else:
        _p(doc, "No continuous numeric columns found.")
    corr = facts.get("top_correlations")
    if corr:
        _p(doc, "Strongest correlations between numeric columns:")
        _table(doc, ["Pair", "Correlation"], list(corr.items()))

    # 7. Categorical summary
    h("Categorical summary")
    cat = g["categorical"][:10]
    if cat:
        _table(
            doc,
            ["Column", "Unique", "Top values (share of rows)"],
            [
                (
                    c,
                    int(df[c].nunique()),
                    ", ".join(f"{k} ({v * 100:.1f}%)" for k, v in df[c].value_counts(normalize=True).head(3).items()),
                )
                for c in cat
            ],
        )
    else:
        _p(doc, "No categorical columns found.")

    # 8. Target analysis
    if target is not None and "target_analysis" in facts:
        h(f"Target analysis: {target}")
        _table(
            doc,
            ["Measure", "Result"],
            [(k.replace("_", " "), _fmt(v)) for k, v in facts["target_analysis"].items() if k != "target"],
        )

    # Prediction model results (only if models were trained on this data and target)
    if ml:
        h("Prediction model results")
        _p(
            doc,
            f"Task: {ml['task']} (target: {ml['target']}). Trained on {ml['n_train']:,} rows and tested on "
            f"{ml['n_test']:,} rows. Best model: {ml['best']}.",
        )
        headers = []
        for r in ml["table"]:
            for k in r:
                if k not in headers:
                    headers.append(k)
        _table(doc, headers, [[r.get(k, "") for k in headers] for r in ml["table"]])
        if ml["task"] == "classification":
            _p(doc, f"Baseline: always guessing the most common class gives accuracy {ml['baseline']}.")
        if ml["features"]:
            _p(doc, "Most important columns (permutation importance):")
            _table(doc, ["Column", "Importance"], ml["features"])
        for cap_text, png in ml["charts"]:
            doc.add_picture(io.BytesIO(png), width=Inches(4.8))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = _p(doc, "")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.add_run(_clean(cap_text)).italic = True
        if ml.get("explanation"):
            _markdown(doc, ml["explanation"])

    # 9. Charts
    charts = make_charts(df, target)
    if charts:
        import matplotlib.pyplot as plt

        h("Charts")
        for i, (caption, fig) in enumerate(charts, 1):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            doc.add_picture(buf, width=Inches(5.2))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = _p(doc, )
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.add_run(_clean(f"Figure {i}: {caption}")).italic = True

    # 10. AI insights
    h("AI insights")
    if insights_text:
        _markdown(doc, insights_text)
    else:
        _p(doc, "AI insights were not included in this report.")

    _p(doc, )
    note = _p(doc, )
    note.add_run(
        "Note: statistics were computed automatically with pandas. AI-written text is based only on those "
        "statistics, but should be reviewed before important decisions."
    ).italic = True

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


# ---------- Streamlit UI ----------
def render_report(df: pd.DataFrame, raw_df: pd.DataFrame | None = None):
    st.caption("Builds a complete Word report for whatever dataset you uploaded.")
    c1, c2 = st.columns(2)
    title = c1.text_input("Report title", "Data Analysis Report")
    subtitle = c2.text_input("Dataset name / subtitle (optional)", "")
    author = c1.text_input("Prepared by (optional)", "")
    target = target_picker(df)
    include_ai = st.checkbox("Include AI executive summary and insights (uses Gemini)", value=True)

    ml_now = st.session_state.get("ml_results")
    if ml_now and ml_now.get("sig") == data_signature(df, target):
        st.success(f"Model results will be included in the report (best model: {ml_now['best']}).")
    else:
        st.info(
            "Tip: model results are not in this report yet. Open the **Prediction** page, pick the same "
            "target column and train the models first, then come back and build the report."
        )

    if st.button("Build report"):
        sig = data_signature(df, target)
        insights_text, exec_summary = None, None

        if include_ai:
            facts = build_facts(df, target)
            with st.spinner("Gemini is writing the summary and insights..."):
                if st.session_state.get("insights_sig") == sig and st.session_state.get("insights_text"):
                    insights_text = st.session_state["insights_text"]
                else:
                    try:
                        insights_text = generate_insights(facts)
                        st.session_state.update(insights_sig=sig, insights_text=insights_text, insights_facts=facts)
                    except Exception as e:
                        st.warning(f"Could not generate insights: {e}")

                if st.session_state.get("exec_sig") == sig and st.session_state.get("exec_summary"):
                    exec_summary = st.session_state["exec_summary"]
                else:
                    try:
                        exec_summary = generate_executive_summary(facts)
                        st.session_state.update(exec_sig=sig, exec_summary=exec_summary)
                    except Exception as e:
                        st.warning(f"Could not generate executive summary: {e}")

        ml = st.session_state.get("ml_results")
        if not ml or ml.get("sig") != sig:
            ml = None
        with st.spinner("Building report..."):
            st.session_state["report_bytes"] = build_report(
                df, raw_df, insights_text, exec_summary, title, subtitle, author, target, ml
            )
        st.session_state["report_preview"] = (exec_summary, insights_text)

    if st.session_state.get("report_bytes"):
        st.success("Report is ready.")
        summary, insights = st.session_state.get("report_preview", (None, None))
        if summary or insights:
            with st.expander("Preview: AI parts of the report"):
                if summary:
                    st.markdown("**Executive summary**")
                    st.write(summary)
                if insights:
                    st.markdown(insights)
        st.download_button(
            "Download report (.docx)",
            st.session_state["report_bytes"],
            file_name="data_analysis_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )