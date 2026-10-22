"""Explicit categorical and continuous plot variables."""
import pandas as pd

MONTH_ORDER = {"January": 1, "February": 2, "March/April": 3.5, "May": 5, "June": 6, "July": 7,
               "August": 8, "September": 9, "September/October": 9.5, "October": 10, "November": 11, "December": 12}


def is_month_variable(series, column):
    values = series.dropna().astype(str)
    return column in {"Month", "Split"} and len(values) > 0 and values.isin(MONTH_ORDER).all()


def default_plot_type(series, column):
    if is_month_variable(series,column) or pd.api.types.is_datetime64_any_dtype(series) or column in {"Date","game_date"}:
        return "Time"
    return "Continuous" if pd.api.types.is_numeric_dtype(series) else "Discrete / factor"


def plot_values(series, mode, column, years=None):
    months = is_month_variable(series,column)
    if mode == "Discrete / factor":
        return series.astype("string").fillna("Missing").astype(str)
    if mode == "Time":
        if months:
            year = pd.to_numeric(years,errors="coerce") if years is not None else pd.Series(2000,index=series.index)
            month = series.astype("string").map(MONTH_ORDER).round().astype("Int64")
            return pd.to_datetime(pd.DataFrame({"year":year,"month":month,"day":1},index=series.index),errors="coerce")
        if pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"{column} contains numbers rather than dates. Use Continuous or Discrete / factor.")
        values = pd.to_datetime(series,format="mixed",errors="coerce")
        if (series.notna() & values.isna()).any():
            raise ValueError(f"{column} has no calendar interpretation. Use Continuous or Discrete / factor.")
        return values
    if months:
        values = series.astype("string").map(MONTH_ORDER)
    elif column == "Half":
        values = series.astype("string").map({"1st Half": 1, "2nd Half": 2})
    elif pd.api.types.is_datetime64_any_dtype(series):
        return series
    else:
        values = pd.to_numeric(series, errors="coerce")
    if (series.notna() & values.isna()).any():
        raise ValueError(f"{column} contains categories without numeric values. Use Discrete / factor.")
    return values


def add_plot_lines(fig, frame, x, y, color, discrete_color, month_order):
    """Connect observations within player/pitch/split series, in axis order."""
    import plotly.graph_objects as go
    identities = ["Name","Player ID","Season","Team","Pitch Type","Pitcher Hand",
                  "Opponent Hand","Home / Away","Half","Pitch Group","Split"]
    groups = [c for c in identities if c in frame and c not in {x,y}]
    if discrete_color and color not in groups:
        groups.append(color)
    traces = list(fig.data)
    colors = {trace.name:trace.marker.color for trace in traces}
    lines = []
    series = frame.groupby(groups,observed=True,dropna=False,sort=False) if groups else [(None,frame)]
    for _, rows in series:
        if len(rows) < 2:
            continue
        if is_month_variable(rows[x],x):
            rows = rows.assign(__order=rows[x].astype(str).map(month_order)).sort_values("__order",kind="stable")
        else:
            rows = rows.sort_values(x,kind="stable")
        group = str(rows[color].iloc[0]) if discrete_color else ""
        shade = colors.get(group, "#888888") if discrete_color or color is None else "#888888"
        lines.append(go.Scatter(x=rows[x],y=rows[y],mode="lines",line=dict(color=shade,width=2),
            legendgroup=group,showlegend=False,hoverinfo="skip"))
    for line in lines:
        fig.add_trace(line)
    fig.data = tuple(fig.data[len(traces):])+tuple(traces)
