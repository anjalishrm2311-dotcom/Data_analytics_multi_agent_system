"""Prediction agent: automatic machine learning (classification / regression) for ANY dataset."""
import io
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from agents.insights_agent import _call_gemini, classify_columns, data_signature, target_picker

# speed presets: how many rows to train on, how many columns to keep, and trees in the forest
SPEEDS = {
    "Fast": {"rows": 10_000, "features": 20, "trees": 60},
    "Balanced": {"rows": 30_000, "features": 30, "trees": 100},
    "Thorough": {"rows": 50_000, "features": 40, "trees": 150},
}
MAX_MISSING = 0.6  # columns with more than 60% missing values are not used as inputs


# ---------- 1. Prepare the data ----------
def _leaky_features(X, y, task, num, cat):
    """Columns that give the answer away (e.g. a code column and its text twin, or a copy of the target)."""
    leaks = []
    if len(X) < 200:
        return leaks
    if task == "classification":
        ys = y.astype(str)
        for c in cat:
            nun = X[c].nunique()
            if nun > 50:
                continue
            pairs = pd.DataFrame({"f": X[c], "y": ys}).drop_duplicates()
            if len(pairs) == nun:  # every value of this column maps to exactly one target class
                leaks.append(c)
    else:
        for c in num:
            r = X[c].corr(y)
            if pd.notna(r) and abs(r) > 0.999:
                leaks.append(c)
    return leaks


def _screen_features(X, y, task, num, cat, k):
    """Quick small random forest to keep only the k most useful columns (keeps training fast)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    Xs = X.sample(min(len(X), 5000), random_state=0)
    ys = y.loc[Xs.index]
    Z = pd.DataFrame(index=Xs.index)
    for c in num:
        Z[c] = Xs[c].fillna(Xs[c].median()).fillna(0)
    for c in cat:
        Z[c] = Xs[c].astype("category").cat.codes
    if task == "classification":
        model = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=0)
    model.fit(Z, ys)
    imp = pd.Series(model.feature_importances_, index=Z.columns).sort_values(ascending=False)
    keep = set(imp.head(k).index)
    return [c for c in X.columns if c in keep]


def prepare(df: pd.DataFrame, target, max_rows=30_000, max_features=30):
    g = classify_columns(df)
    if target in g["numeric"]:
        task = "regression"
    elif target in g["categorical"]:
        task = "classification"
    else:
        raise ValueError(
            "This target column is not suitable (it is text, an ID, constant or a date). "
            "Pick a numeric column (regression) or a column with a few categories (classification)."
        )

    base_drop = set(g["id_like"] + g["text"] + g["constant"] + g["datetime"]) | {target}
    reasons = {"IDs, long text, dates or constant values": [c for c in df.columns if c in base_drop and c != target]}
    features = [c for c in df.columns if c not in base_drop]

    miss = df[features].isna().mean() if features else pd.Series(dtype=float)
    sparse = [c for c in features if miss[c] > MAX_MISSING]
    reasons[f"mostly empty (over {int(MAX_MISSING * 100)}% missing)"] = sparse
    features = [c for c in features if c not in sparse]
    if not features:
        raise ValueError("No usable input columns are left (they are IDs, text, dates, constants or mostly empty).")

    data = df[features + [target]].dropna(subset=[target])
    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
        sampled = True
    data = data.reset_index(drop=True)

    X, y = data[features].copy(), data[target]
    num = [c for c in features if c in g["numeric"]]
    cat = [c for c in features if c not in num]
    for c in cat:  # treat missing as its own category so encoders never see mixed types
        X[c] = X[c].where(X[c].notna(), "missing").astype(str)

    leaks = _leaky_features(X, y, task, num, cat)
    reasons["too closely tied to the target (a copy or a perfect giveaway)"] = leaks
    if leaks:
        X = X.drop(columns=leaks)
        num = [c for c in num if c not in leaks]
        cat = [c for c in cat if c not in leaks]
    if X.shape[1] == 0:
        raise ValueError("No usable input columns are left after removing columns that give the answer away.")

    if X.shape[1] > max_features:
        keep = _screen_features(X, y, task, num, cat, max_features)
        cut = [c for c in X.columns if c not in keep]
        reasons[f"less useful (only the top {max_features} columns are kept)"] = cut
        X = X[keep]
        num = [c for c in num if c in keep]
        cat = [c for c in cat if c in keep]

    reasons = {k: [str(c) for c in v] for k, v in reasons.items() if v}
    return task, X, y, num, cat, reasons, sampled


def _preprocessor(num, cat):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=10)),
            ]), cat)
        )
    return ColumnTransformer(parts, sparse_threshold=0)  # dense output works with every model


def _models(task, trees):
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=trees, n_jobs=-1, class_weight="balanced", random_state=42),
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        }
    return {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=trees, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }


def _score(task, y_true, y_pred) -> dict:
    from sklearn import metrics as m

    if task == "classification":
        return {
            "Accuracy": round(float(m.accuracy_score(y_true, y_pred)), 3),
            "Precision": round(float(m.precision_score(y_true, y_pred, average="weighted", zero_division=0)), 3),
            "Recall": round(float(m.recall_score(y_true, y_pred, average="weighted", zero_division=0)), 3),
            "F1": round(float(m.f1_score(y_true, y_pred, average="weighted", zero_division=0)), 3),
        }
    return {
        "R²": round(float(m.r2_score(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(m.mean_squared_error(y_true, y_pred))), 3),
        "MAE": round(float(m.mean_absolute_error(y_true, y_pred)), 3),
    }


def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------- 2. Train and compare ----------
def train(df: pd.DataFrame, target, sig: str, speed: str = "Balanced", progress=None) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    cfg = SPEEDS.get(speed, SPEEDS["Balanced"])
    t_start = time.time()

    def step(frac, text):
        if progress:
            progress(frac, text)

    step(0.02, "Preparing the data (removing unusable columns, picking the best inputs)...")
    task, X, y, num, cat, reasons, sampled = prepare(df, target, cfg["rows"], cfg["features"])

    strat = None
    class_names = None
    if task == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("The target has only one class, so there is nothing to predict.")
        # encode classes as 0..k-1 (works with every model and class_weight); names are mapped back for display
        class_names = sorted(y.unique())
        y = y.map({c: i for i, c in enumerate(class_names)})
        strat = y if counts.min() >= 2 else None
    if len(X) < 30:
        raise ValueError("Not enough rows with a known target (need at least 30).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    models = _models(task, cfg["trees"])
    rows, fitted = [], {}
    for i, (name, model) in enumerate(models.items()):
        step(0.10 + 0.55 * i / len(models), f"Training {name} ({i + 1} of {len(models)})...")
        t0 = time.time()
        pipe = Pipeline([("prep", _preprocessor(num, cat)), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
        except Exception as e:
            rows.append({"Model": name, "Note": f"failed: {str(e)[:80]}"})
            continue
        rows.append({"Model": name, **_score(task, y_test, pred), "Train time (s)": round(time.time() - t0, 1)})
        fitted[name] = (pipe, pred)
    if not fitted:
        raise ValueError("All models failed to train. Check the data and the target column.")

    main = "F1" if task == "classification" else "R²"
    score_of = {r["Model"]: r[main] for r in rows if main in r}
    best = max(score_of, key=score_of.get)
    best_pipe, best_pred = fitted[best]
    baseline = float(y_test.value_counts(normalize=True).iloc[0]) if task == "classification" else 0.0

    step(0.70, "Finding the most important columns...")
    features = []
    imp = None
    try:
        Xs = X_test.sample(min(len(X_test), 1000), random_state=42)
        ys = y_test.loc[Xs.index]
        r = permutation_importance(
            best_pipe, Xs, ys, n_repeats=2, random_state=42,
            scoring="f1_weighted" if task == "classification" else "r2",
        )
        imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        features = [(str(k), round(float(v), 4)) for k, v in imp.head(10).items()]
    except Exception:
        pass

    step(0.90, "Drawing charts...")
    charts = []
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    pd.Series(score_of).plot(kind="bar", ax=ax)
    ax.set_ylabel(main)
    ax.set_title(f"Model comparison ({main})", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    charts.append(("Model comparison", _png(fig)))

    if task == "classification":
        labels = sorted(y_test.unique())
        if len(labels) <= 15:
            cm = confusion_matrix(y_test, best_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([class_names[l] for l in labels], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([class_names[l] for l in labels], fontsize=8)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion matrix ({best})", fontsize=10)
            fig.tight_layout()
            charts.append(("Confusion matrix of the best model", _png(fig)))
    else:
        k = min(len(y_test), 1000)
        idx = np.random.default_rng(42).choice(len(y_test), k, replace=False)
        yt, yp = np.asarray(y_test)[idx], np.asarray(best_pred)[idx]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(yt, yp, s=8, alpha=0.5)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs actual ({best})", fontsize=10)
        fig.tight_layout()
        charts.append(("Predicted vs actual (best model)", _png(fig)))

    if imp is not None:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        imp.head(10).iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_title("Feature importance (permutation)", fontsize=10)
        fig.tight_layout()
        charts.append(("Feature importance", _png(fig)))

    defaults, options = {}, {}
    for c in X.columns:
        if c in cat:
            options[c] = [str(v) for v in X[c].value_counts().head(30).index]
            defaults[c] = options[c][0]
        else:
            defaults[c] = float(X[c].median())

    step(1.0, "Done.")
    return {
        "sig": sig,
        "task": task,
        "target": str(target),
        "speed": speed,
        "seconds": round(time.time() - t_start, 1),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sampled": sampled,
        "n_features": int(X.shape[1]),
        "dropped_reasons": reasons,
        "dropped": [c for v in reasons.values() for c in v],
        "main_metric": main,
        "table": rows,
        "best": best,
        "baseline": round(baseline, 3),
        "features": features,
        "charts": charts,
        "pipeline": best_pipe,
        "columns": list(X.columns),
        "num_cols": num,
        "cat_cols": cat,
        "defaults": defaults,
        "options": options,
        "explanation": None,
        "class_names": class_names,
    }


def explain_results(res: dict) -> str:
    facts = {
        "task": res["task"],
        "target": res["target"],
        "rows_used_for_training": res["n_train"],
        "rows_used_for_testing": res["n_test"],
        "input_columns_used": res["n_features"],
        "model_scores": res["table"],
        "best_model": res["best"],
        "baseline": (
            f"always predicting the most common class gives accuracy {res['baseline']}"
            if res["task"] == "classification"
            else "predicting the average gives R² = 0"
        ),
        "top_features_permutation_importance": res["features"],
        "columns_not_used_and_why": res["dropped_reasons"],
    }
    prompt = (
        "You are a data scientist explaining model results to a non-technical reader. Use ONLY the facts below; "
        "never invent a number. Write 5-6 short markdown bullets: what the model predicts, which model is best and "
        "how good it is compared with the baseline, which columns matter most, one honest caution (importance is "
        "not causation; small or imbalanced data), and 1-2 next steps. No headings.\n\n"
        f"FACTS (JSON):\n{json.dumps(facts, indent=2, default=str)}"
    )
    return _call_gemini(prompt)


# ---------- 3. Streamlit UI ----------
def render_prediction(df: pd.DataFrame):
    st.caption(
        "The app learns from the rows of your data and then predicts one column (the target) for a new row. "
        "Pick the column you want to predict."
    )
    target = target_picker(df)
    if target is None:
        st.info("Choose a target column above (the value you want to predict).")
        return

    speed = st.selectbox(
        "Training speed",
        list(SPEEDS),
        index=1,
        help="Fast uses fewer rows and columns (about 10,000 rows). Thorough uses more (up to 50,000 rows) and takes longer.",
    )

    sig = data_signature(df, target)
    res = st.session_state.get("ml_results")
    if res and res.get("sig") != sig:
        res = None

    if st.button("Train models"):
        bar = st.progress(0.0, text="Starting...")
        try:
            res = train(df, target, sig, speed, progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
            st.session_state["ml_results"] = res
        except ValueError as e:
            st.error(str(e))
            res = None
        except ImportError:
            st.error("scikit-learn is missing. Run: python -m pip install scikit-learn")
            res = None
        except Exception as e:
            st.error(f"Training failed: {e}")
            res = None
        finally:
            bar.empty()

    if not res:
        st.info("Click **Train models** to start.")
        return

    kind = "classification" if res["task"] == "classification" else "regression"
    if res["task"] == "classification":
        names = res["class_names"]
        shown = ", ".join(names[:8]) + (f" ... ({len(names)} classes)" if len(names) > 8 else "")
        st.markdown(f"**Predicting:** `{res['target']}` — which one of these values a row has: {shown}")
    else:
        st.markdown(f"**Predicting:** `{res['target']}` — a number")
    msg = (
        f"Task: **{kind}**. Trained on {res['n_train']:,} rows, tested on {res['n_test']:,} rows, "
        f"using {res['n_features']} input columns ({res['speed']} mode, {res['seconds']:.0f} seconds)."
    )
    if res["sampled"]:
        msg += " Large data: a random sample of rows was used to keep it fast."
    st.write(msg)
    if res["dropped_reasons"]:
        with st.expander("Columns not used, and why"):
            for reason, cols in res["dropped_reasons"].items():
                st.markdown(f"**{reason}** ({len(cols)}): " + ", ".join(cols[:25]) + (" ..." if len(cols) > 25 else ""))

    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(res["table"]))
    st.success(f"Best model: {res['best']} ({res['main_metric']} = "
               f"{next(r[res['main_metric']] for r in res['table'] if r['Model'] == res['best'])})")
    if res["task"] == "classification":
        st.caption(f"Baseline: always guessing the most common class gives accuracy {res['baseline']}.")
    else:
        st.caption("Baseline: predicting the average value gives R² = 0.")

    cols = st.columns(2)
    for i, (_, png) in enumerate(res["charts"]):
        cols[i % 2].image(png)

    st.subheader("AI explanation")
    if st.button("Explain results with AI"):
        with st.spinner("Gemini is explaining the results..."):
            try:
                res["explanation"] = explain_results(res)
            except Exception as e:
                st.error(f"Gemini error: {e}")
    if res.get("explanation"):
        st.markdown(res["explanation"])

    st.subheader("Try a prediction")
    st.caption(f"Fill in the values of one example row and the model predicts `{res['target']}` for it.")
    by_name = {str(c): c for c in res["columns"]}
    feats = [by_name[f] for f, _ in res["features"] if f in by_name][:8] or res["columns"][:8]
    with st.form("ml_predict_form"):
        vals = {}
        grid = st.columns(2)
        for i, f in enumerate(feats):
            box = grid[i % 2]
            if f in res["cat_cols"]:
                opts = res["options"][f]
                vals[f] = box.selectbox(str(f), opts, key=f"mlin_{res['sig']}_{f}")
            else:
                vals[f] = box.number_input(str(f), value=float(res["defaults"][f]), key=f"mlin_{res['sig']}_{f}")
        go = st.form_submit_button("Predict")
    st.caption("Columns you do not see here use a typical value (median or most common).")
    if go:
        row = {c: res["defaults"][c] for c in res["columns"]}
        row.update(vals)
        X_new = pd.DataFrame([row], columns=res["columns"])
        for c in res["cat_cols"]:
            X_new[c] = X_new[c].astype(str)
        pipe = res["pipeline"]
        pred = pipe.predict(X_new)[0]
        if res["task"] == "regression":
            st.success(f"Predicted {res['target']}: {float(pred):,.3f}")
        else:
            names = res["class_names"]
            st.success(f"Predicted {res['target']}: {names[int(pred)]}")
            if hasattr(pipe, "predict_proba"):
                proba = pd.Series(pipe.predict_proba(X_new)[0], index=[names[int(c)] for c in pipe.classes_])
                st.write("Class probabilities:")
                st.dataframe(proba.sort_values(ascending=False).head(5).rename("probability").round(3))

    buf = io.BytesIO()
    joblib.dump(
        {"pipeline": res["pipeline"], "features": res["columns"], "target": res["target"], "task": res["task"],
         "class_names": res["class_names"]}, buf
    )
    st.download_button("Download trained model (.joblib)", buf.getvalue(), file_name="best_model.joblib")