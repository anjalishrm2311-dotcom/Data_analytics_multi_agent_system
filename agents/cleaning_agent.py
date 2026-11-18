import pandas as pd


def cleaning_agent(df: pd.DataFrame):
    """
    Handles missing values, duplicates, and basic data type issues.
    Returns the cleaned dataframe plus a report of what was changed.
    """

    report = {}

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # 2. Handle missing values column by column
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_values = df[col].mode()
            fill_value = mode_values[0] if not mode_values.empty else ""
            df[col] = df[col].fillna(fill_value)

        report[f"{col}_missing_filled"] = int(missing_count)

    # 3. Fix common date columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Strip extra whitespace from text columns
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    return df, report