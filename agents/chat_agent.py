import ast
import builtins
import os
import re
import time

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Models are tried in order. Override with GEMINI_MODEL in .env (comma separated).
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TIMEOUT_MS = 30_000
RETRIES = 2
EXEC_SECONDS = 20  # generated code is stopped after this many seconds


def _models():
    env = os.environ.get("GEMINI_MODEL", "").strip()
    return [m.strip() for m in env.split(",") if m.strip()] or DEFAULT_MODELS


def _code_enabled() -> bool:
    """Running generated code is turned off in public deployments (PUBLIC_MODE=1)."""
    return os.environ.get("PUBLIC_MODE", "0") != "1"


NO_CODE_NOTE = """IMPORTANT OVERRIDE: running code is DISABLED in this deployment. Never output code blocks.
Answer in plain text using only the DATASET SUMMARY. If a question needs calculations or a chart that the
summary cannot answer, say briefly that this public demo cannot run calculations and suggest the Insights,
Prediction or Reports pages.

"""


# ============================================================
# DATASET SUMMARY
# ============================================================

def build_dataset_summary(df: pd.DataFrame, max_cols: int = 30, raw_df: pd.DataFrame = None) -> str:
    """Text summary of the dataset, sent to Gemini as context.

    The real row/column counts and ALL column names are always included. Detailed statistics,
    correlations and sample rows cover only the first `max_cols` columns to keep the prompt small.
    """
    total_cols = df.shape[1]
    parts = [f"Rows: {df.shape[0]}, Columns: {total_cols}"]

    if raw_df is not None and raw_df is not df:
        try:
            parts.append(
                "BEFORE cleaning (original upload): "
                f"{raw_df.shape[0]} rows, {raw_df.shape[1]} columns, {int(raw_df.isnull().sum().sum())} missing cells.\n"
                "AFTER cleaning (the data you are analysing): "
                f"{df.shape[0]} rows, {df.shape[1]} columns, {int(df.isnull().sum().sum())} missing cells."
            )
        except Exception:
            pass

    missing = df.isnull().sum()
    lines = [f"{c} ({t}, {int(m)} missing)" for c, t, m in zip(df.columns, df.dtypes.astype(str), missing)]
    shown = lines[:300]
    extra = f"\n... and {len(lines) - 300} more columns" if len(lines) > 300 else ""
    parts.append(f"All {total_cols} columns (name, dtype, missing count):\n" + "\n".join(shown) + extra)

    try:
        dups = int(df.duplicated().sum())
    except Exception:
        dups = int(df.astype(str).duplicated().sum())
    parts.append(f"Duplicate rows: {dups}")

    sub = df.iloc[:, :max_cols]
    if total_cols > max_cols:
        parts.append(
            f"NOTE: the statistics, correlations and sample rows below cover only the first {max_cols} "
            f"of {total_cols} columns. The column list above is complete."
        )
    try:
        parts.append("Descriptive statistics:\n" + sub.describe(include="all").T.to_string())
    except Exception:
        pass
    num = sub.select_dtypes(include="number")
    if num.shape[1] >= 2:
        parts.append("Correlation (numeric columns):\n" + num.corr().round(3).to_string())
    parts.append("First 5 rows:\n" + sub.head(5).to_string())
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are a data analyst assistant working on ONE pandas DataFrame named `df`.

You have two ways to answer:

1. If the question can be answered from the DATASET SUMMARY below (row counts,
   dtypes, missing values, mean/min/max, correlations, explaining columns,
   advice), answer in plain text. Use exact numbers from the summary.

2. If the question needs real calculation on the data (group-wise results,
   filtering, top N, counts per category, percentages by group) or asks for a
   chart, reply with ONE short sentence explaining what you will do, followed
   by exactly one ```python code block.

Rules for the code block:
- Use only the variable `df` plus `pd`, `np` and `plt`. They are already
  available. Do NOT write import statements.
- Use the exact column names from the summary.
- Put the final answer in a variable named `result` (a DataFrame, Series or
  a single value).
- For a chart, draw it with plt (one figure, with a title and axis labels)
  and do not call plt.show(). You may also set `result` to the numbers behind it.
- Never read or write files, never access the network, never modify `df`
  in place (work on a copy if you need to change it).
- The code runs in a restricted sandbox. Allowed: plain Python (if, for, while,
  def, lambda, comprehensions, f-strings), pandas, numpy and plt. NOT allowed:
  import, try/except, with, class, str.format() (use f-strings instead),
  df.query() / df.eval() (use boolean indexing instead), df.style, saving
  figures or files, and any name or attribute that starts with an underscore.
- Keep the code short.

Never invent numbers. Reply in the same language the user writes in.

DATASET SUMMARY:
"""


# ============================================================
# GEMINI CALL (retries + fallback models)
# ============================================================

def _check_ready():
    if genai is None:
        return "google-genai is not installed in the Python running Streamlit."
    if not os.environ.get("GEMINI_API_KEY"):
        return "GEMINI_API_KEY was not found. Check the .env file name, location and restart."
    return None


def _is_busy(err: Exception) -> bool:
    text = str(err).lower()
    return any(s in text for s in ("503", "unavailable", "429", "resource_exhausted",
                                   "overloaded", "timed out", "timeout", "deadline"))


def _generate(contents, system) -> str:
    """Returns the model text, or raises RuntimeError with a readable message."""
    config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=2048)
    last_error = None
    for model in _models():
        for attempt in range(RETRIES + 1):
            try:
                client = genai.Client(
                    api_key=os.environ["GEMINI_API_KEY"],
                    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
                )
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                text = (response.text or "").strip()
                if text:
                    return text
                last_error = "empty answer"
                break
            except Exception as e:
                last_error = e
                if _is_busy(e) and attempt < RETRIES:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise RuntimeError(
        "Gemini is not responding right now. "
        f"Last error: {last_error}. Try again in a minute, or set GEMINI_MODEL in .env."
    )


# ============================================================
# SANDBOXED CODE RUNNER
#
# Layers: (1) the code is parsed and checked as a syntax tree (not by
# searching for words), (2) it runs with a tiny set of built-in functions,
# (3) it works on a copy of the data, (4) loops are stopped after a time limit.
# This greatly reduces risk, but it is NOT a full OS-level sandbox:
# run the app only on your own machine unless the code runs in a container.
# ============================================================

class _CodeTimeout(BaseException):
    """BaseException so generated code or pandas cannot swallow it by accident."""


# syntax that generated code may not use
_BLOCKED_NODES = tuple(
    getattr(ast, n) for n in (
        "Import", "ImportFrom", "ClassDef", "Global", "Nonlocal", "Try", "TryStar",
        "With", "AsyncWith", "AsyncFor", "AsyncFunctionDef", "Await", "Yield", "YieldFrom",
    ) if hasattr(ast, n)
)

# names that must never appear
_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit",
    "os", "sys", "subprocess", "importlib", "builtins", "shutil", "pathlib", "pickle",
    "ctypes", "socket", "urllib", "requests", "http",
}

# attributes that must never appear (file/network/system access, ways to reach other modules)
_BLOCKED_ATTRS = {
    "os", "sys", "io", "subprocess", "importlib", "builtins", "shutil", "pathlib", "pickle",
    "ctypes", "socket", "urllib", "requests", "http", "matplotlib",
    "lib", "ctypeslib", "f2py", "testing", "compat", "util", "core",
    "eval", "query", "format", "format_map", "style",
    "open", "imread", "imsave", "memmap", "DataSource", "fromregex", "fromfile", "tofile",
    "genfromtxt", "system", "popen", "remove", "rmdir", "unlink", "mkdir", "makedirs",
    "rename", "listdir", "walk", "getenv", "environ", "chdir", "urlopen", "urlretrieve",
}
# any attribute starting with these is blocked (read_csv, to_csv, savefig, loadtxt, dump...)
_BLOCKED_PREFIXES = ("read_", "to_", "save", "load", "dump", "write")
# ...except these harmless conversions
_ALLOWED_TO = {
    "to_frame", "to_numpy", "to_list", "to_dict", "to_datetime", "to_numeric", "to_period",
    "to_timestamp", "to_series", "to_pydatetime", "to_timedelta", "to_flat_index",
}


def _safe_range(*args):
    r = range(*args)
    if len(r) > 10_000_000:
        raise ValueError("range() is too large for chat analysis.")
    return r


_SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in ("len", "min", "max", "sum", "sorted", "round", "abs", "int", "float", "str",
              "list", "dict", "set", "tuple", "enumerate", "zip", "bool", "any", "all",
              "reversed", "isinstance", "print", "map", "filter", "pow", "divmod")
}
_SAFE_BUILTINS["range"] = _safe_range
# numpy/pandas sometimes import modules lazily from inside their own C code, and that lookup
# uses these builtins. Generated code can NOT reach this name (names starting with "__" and
# import statements are blocked by _validate), so adding it does not give the code any new power.
_SAFE_BUILTINS["__import__"] = builtins.__import__


def _blocked(message: str, node) -> ValueError:
    line = getattr(node, "lineno", None)
    where = f" (line {line})" if line else ""
    return ValueError(f"Blocked: {message}{where}.")


def _validate(tree: ast.AST):
    """Raises ValueError if the generated code uses anything that is not allowed."""
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise _blocked(f"'{type(node).__name__}' is not allowed", node)
        if isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
                raise _blocked(f"the name '{node.id}' is not allowed", node)
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                raise _blocked(f"the attribute '{a}' is not allowed", node)
            if a in _BLOCKED_ATTRS:
                raise _blocked(f"the attribute '{a}' is not allowed", node)
            if a.startswith(_BLOCKED_PREFIXES) and a not in _ALLOWED_TO:
                raise _blocked(f"the attribute '{a}' is not allowed (files and exports are blocked)", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise _blocked("text containing '__' is not allowed", node)


class _AddTicks(ast.NodeTransformer):
    """Inserts a time check into every loop and comprehension so runaway code is stopped."""

    @staticmethod
    def _call():
        return ast.Call(func=ast.Name(id="__tick__", ctx=ast.Load()), args=[], keywords=[])

    def visit_For(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.Expr(value=self._call()))
        return node

    visit_While = visit_For

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.ifs.append(self._call())
        return node


def _make_ticker(seconds: float):
    deadline = time.monotonic() + seconds

    def tick():
        if time.monotonic() > deadline:
            raise _CodeTimeout()
        return True

    return tick


def _extract_code(text: str):
    """Splits the model reply into (explanation, code or None)."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return text.strip(), None
    explanation = (text[:m.start()] + text[m.end():]).strip()
    return explanation, m.group(1).strip()


def _run_code(code: str, df: pd.DataFrame):
    """Runs the generated code in the sandbox. Returns (result, figure). Raises on error."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"The generated code has a syntax error: {e.msg} (line {e.lineno}).")
    _validate(tree)
    tree = ast.fix_missing_locations(_AddTicks().visit(tree))
    compiled = compile(tree, "<generated>", "exec")

    env = {
        "__builtins__": _SAFE_BUILTINS,
        "__tick__": _make_ticker(EXEC_SECONDS),
        "pd": pd, "np": np, "plt": plt,
        "df": df.copy(),  # the loaded data can never be changed by chat code
    }
    if plt is not None:
        plt.close("all")

    try:
        exec(compiled, env)
    except _CodeTimeout:
        raise TimeoutError(f"The code ran for more than {EXEC_SECONDS} seconds and was stopped.")

    fig = None
    if plt is not None and plt.get_fignums():
        fig = plt.gcf()
    return env.get("result"), fig


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def ask_data(df: pd.DataFrame, question: str, history: list, raw_df: pd.DataFrame = None) -> dict:
    """
    Answers a question about df.
    Returns {"text": str, "code": str|None, "table": DataFrame|None, "fig": Figure|None}.
    """
    out = {"text": "", "code": None, "table": None, "fig": None}

    problem = _check_ready()
    if problem:
        out["text"] = problem
        return out

    system = ("" if _code_enabled() else NO_CODE_NOTE) + SYSTEM_PROMPT + build_dataset_summary(df, raw_df=raw_df)

    contents = []
    for m in history[-10:]:
        role = "user" if m["role"] == "user" else "model"
        if m.get("content"):
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    try:
        reply = _generate(contents, system)
    except RuntimeError as e:
        out["text"] = str(e)
        return out

    explanation, code = _extract_code(reply)
    if code is None:
        out["text"] = reply
        return out

    if not _code_enabled():  # safety net in case the model still writes code
        out["text"] = (
            explanation
            + "\n\n_Running code is turned off in this public demo. "
            "Use the Insights, Prediction and Reports pages for analysis and charts._"
        ).strip()
        return out

    # Run the code. If it fails, give the model the error once and let it fix it.
    for attempt in range(2):
        try:
            result, fig = _run_code(code, df)
            out["code"], out["fig"] = code, fig
            out["text"] = explanation
            if isinstance(result, pd.DataFrame):
                out["table"] = result.head(200)
            elif isinstance(result, pd.Series):
                out["table"] = result.head(200).to_frame()
            elif result is not None:
                out["text"] = (explanation + f"\n\n**Result:** {result}").strip()
            if fig is None and result is None:
                out["text"] = (explanation + "\n\n(The code ran but produced no result.)").strip()
            return out
        except Exception as e:
            if attempt == 1:
                out["code"] = code
                out["text"] = f"{explanation}\n\nThe generated code failed: {e}".strip()
                return out
            contents.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            contents.append(types.Content(role="user", parts=[types.Part(
                text=f"That code failed with this error: {e}. Reply again with one short "
                     "sentence and a corrected ```python code block that follows the sandbox rules."
            )]))
            try:
                reply = _generate(contents, system)
            except RuntimeError as err:
                out["text"] = str(err)
                return out
            explanation, new_code = _extract_code(reply)
            if new_code is None:
                out["text"] = reply
                return out
            code = new_code

    return out