from pathlib import Path

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium

from src.io import load_geo, load_population, merge_population


# -------------------------------
# Page config MUST be first Streamlit call
# -------------------------------
st.set_page_config(page_title="Egypt Governorates Atlas", layout="wide")


# -------------------------------
# Paths (single source of truth)
# -------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "data" / "tables"


def pick_geojson(data_dir: Path) -> Path:
    geo_files = sorted(data_dir.glob("*.geojson"))
    if not geo_files:
        raise FileNotFoundError(f"No .geojson found in: {data_dir}")
    return next((p for p in geo_files if "govern" in p.name.lower()), geo_files[0])


GEO_PATH = pick_geojson(DATA)
POP_PATH = DATA / "population.csv"
IND_PATH = TABLES / "indicators.csv"


# -------------------------------
# Data status panel (safe)
# -------------------------------
with st.sidebar.expander("Data status", expanded=False):
    st.write("Processed data folder:", str(DATA))
    st.write("GeoJSON:", GEO_PATH.name, "✅" if GEO_PATH.exists() else "❌")
    st.write("Population CSV:", POP_PATH.name, "✅" if POP_PATH.exists() else "❌")
    st.write("Indicators CSV:", IND_PATH.name, "✅" if IND_PATH.exists() else "❌")


# -------------------------------
# UI CSS tweaks (legend font size)
# -------------------------------
st.markdown(
    """
<style>
.leaflet-control .legend { font-size: 11px !important; line-height: 1.2 !important; }
.leaflet-control .legend-title { font-size: 12px !important; font-weight: 600 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------
# Helpers
# -------------------------------
def to_num(x):
    """Safe scalar -> numeric (float) or NaN (avoids pd.to_numeric scalar TypeError)."""
    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]


def compute_stats(df: pd.DataFrame, indicator: str) -> dict:
    s = pd.to_numeric(df[indicator], errors="coerce")
    s_non = s.dropna()
    if len(s_non) == 0:
        return {"mean": None, "min": None, "max": None, "ranks": None, "n": 0}
    mean_val = float(s_non.mean())
    min_val = float(s_non.min())
    max_val = float(s_non.max())
    ranks = s_non.rank(ascending=False, method="min")  # 1 = highest
    return {"mean": mean_val, "min": min_val, "max": max_val, "ranks": ranks, "n": int(len(s_non))}


def wow_sentence(lang: str, indicator_col: str, percentile: float) -> str:
    if percentile is None:
        return "—"
    top10 = percentile >= 90
    top25 = percentile >= 75
    mid = 25 <= percentile < 75

    if indicator_col == "compactness":
        if lang == "English":
            if top10:
                return "Extremely compact shape (Top 10%)."
            if top25:
                return "More compact than most (Top 25%)."
            if mid:
                return "Moderately compact (Middle range)."
            return "Among the least compact shapes (Bottom 25%)."
        else:
            if top10:
                return "شكل شديد التماسك (ضمن أعلى 10%)."
            if top25:
                return "أكثر تماسكًا من معظم المحافظات (أعلى 25%)."
            if mid:
                return "تماسك متوسط (ضمن النطاق الأوسط)."
            return "من أقل الأشكال تماسكًا (أدنى 25%)."

    if lang == "English":
        if top10:
            return "Top 10% nationally."
        if top25:
            return "Top 25% nationally."
        if mid:
            return "Around the national middle range."
        return "Bottom 25% nationally."
    else:
        if top10:
            return "ضمن أعلى 10% على مستوى مصر."
        if top25:
            return "ضمن أعلى 25% على مستوى مصر."
        if mid:
            return "ضمن النطاق المتوسط على مستوى مصر."
        return "ضمن أدنى 25% على مستوى مصر."


def top_bottom_table(df: pd.DataFrame, indicator: str, n: int = 5):
    t = df[["gov_id", "name_en", "name_ar", indicator]].copy()
    t[indicator] = pd.to_numeric(t[indicator], errors="coerce")
    t = t.dropna(subset=[indicator])
    top = t.sort_values(indicator, ascending=False).head(n)
    bottom = t.sort_values(indicator, ascending=True).head(n)
    return top, bottom


def format_value(indicator_col: str, x):
    if pd.isna(x):
        return "—"
    if indicator_col == "compactness":
        return f"{float(x):.3f}"
    return f"{float(x):,.1f}"

def dedupe_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    GeoPandas cannot contain duplicated column names.
    Keep the first occurrence and drop the rest.
    """
    return gdf.loc[:, ~gdf.columns.duplicated()].copy()

# -------------------------------
# Data loading (ONE function, always returns)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(geo_path: str, pop_path: str, ind_path: str, year: int):
    # Base geo
    gdf = load_geo(geo_path)
    gdf = dedupe_columns(gdf)

    # Population (optional)
    if Path(pop_path).exists():
        pop = load_population(pop_path)
        gdf = merge_population(gdf, pop, year=year)
        gdf = dedupe_columns(gdf)
    else:
        if "population" not in gdf.columns:
            gdf["population"] = 0

    # Rename population -> population_2023 safely
    if "population" in gdf.columns:
        if "population_2023" in gdf.columns and "population" != "population_2023":
            # If both exist, prefer population_2023 and drop population
            gdf = gdf.drop(columns=["population"])
        else:
            gdf = gdf.rename(columns={"population": "population_2023"})
    else:
        if "population_2023" not in gdf.columns:
            gdf["population_2023"] = 0

    gdf = dedupe_columns(gdf)

    # Indicators table (optional)
    if Path(ind_path).exists():
        indicators = pd.read_csv(ind_path)

        # Ensure gov_id types align
        gdf["gov_id"] = pd.to_numeric(gdf["gov_id"], errors="coerce").fillna(-1).astype(int)
        indicators["gov_id"] = pd.to_numeric(indicators["gov_id"], errors="coerce").fillna(-1).astype(int)

        # Avoid duplicate indicator columns if repeated runs or overlaps
        overlap = set(gdf.columns).intersection(set(indicators.columns)) - {"gov_id"}
        if overlap:
            indicators = indicators.drop(columns=list(overlap))

        gdf = gdf.merge(indicators, on="gov_id", how="left")
        gdf = dedupe_columns(gdf)

    # Ensure CRS for web mapping
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    return gdf



DEFAULT_YEAR = 2024
gdf = load_data(str(GEO_PATH), str(POP_PATH), str(IND_PATH), DEFAULT_YEAR)

def dedupe_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    GeoPandas cannot handle duplicated column names.
    This keeps the first occurrence and drops the rest.
    """
    return gdf.loc[:, ~gdf.columns.duplicated()].copy()

# -------------------------------
# Guardrails: ensure required columns exist
# -------------------------------
for col in ["area_km2", "perimeter_km", "compactness", "centroid_lat", "centroid_lon"]:
    if col not in gdf.columns:
        gdf[col] = 0.0

if "population_2023" not in gdf.columns:
    gdf["population_2023"] = 0

# If your dataset doesn't contain these name fields, create fallbacks
if "name_en" not in gdf.columns:
    for c in ["gov_name", "NAME_EN", "ADM1_EN", "name"]:
        if c in gdf.columns:
            gdf["name_en"] = gdf[c].astype(str)
            break
    else:
        gdf["name_en"] = gdf["gov_id"].astype(str)

if "name_ar" not in gdf.columns:
    for c in ["NAME_AR", "ADM1_AR", "name_ar"]:
        if c in gdf.columns:
            gdf["name_ar"] = gdf[c].astype(str)
            break
    else:
        gdf["name_ar"] = gdf["name_en"].astype(str)

# Detect if population is effectively empty (SUPER SAFE)
if "population_2023" in gdf.columns:
    pop_col = gdf["population_2023"]

    # If duplicate columns exist, gdf["population_2023"] can become a DataFrame.
    # This forces a single Series:
    if isinstance(pop_col, pd.DataFrame):
        pop_col = pop_col.iloc[:, 0]

    pop_series = pd.to_numeric(pd.Series(pop_col), errors="coerce").fillna(0)
else:
    pop_series = pd.Series(0, index=gdf.index, dtype="float64")

population_is_empty = (pop_series <= 0).all()



# -------------------------------
# Metric descriptions (EN/AR)
# -------------------------------
metric_help = {
    "population_2023": {
        "en": "Estimated residents (demo). Replace with official data when available.",
        "ar": "تقدير عدد السكان (عرض تجريبي). استبدله بالبيانات الرسمية لاحقًا.",
    },
    "area_km2": {"en": "Governorate land area in km².", "ar": "مساحة المحافظة كم²."},
    "perimeter_km": {"en": "Boundary length in km.", "ar": "طول الحدود كم."},
    "compactness": {"en": "Shape compactness (0–1).", "ar": "تماسك الشكل (0–1)."},
}


# -------------------------------
# Sidebar
# -------------------------------
lang = st.sidebar.radio("Language / اللغة", options=["English", "العربية"], index=0)
display_col = "name_en" if lang == "English" else "name_ar"

if lang == "English":
    indicator_options = {
        "Population (2023)": "population_2023",
        "Area (km²)": "area_km2",
        "Perimeter (km)": "perimeter_km",
        "Compactness Index": "compactness",
    }
else:
    indicator_options = {
        "عدد السكان (2023)": "population_2023",
        "المساحة (كم²)": "area_km2",
        "المحيط (كم)": "perimeter_km",
        "مؤشر التماسك": "compactness",
    }

indicator_label = st.sidebar.selectbox(
    "Indicator" if lang == "English" else "المؤشر",
    options=list(indicator_options.keys()),
)
indicator_col = indicator_options[indicator_label]
if indicator_col not in gdf.columns:
    indicator_col = "population_2023"

desc = metric_help.get(indicator_col, {})
st.sidebar.caption(desc.get("en", "") if lang == "English" else desc.get("ar", ""))

# Ensure gov_id is int everywhere (important!)
gdf["gov_id"] = pd.to_numeric(gdf["gov_id"], errors="coerce").fillna(-1).astype(int)

# Session defaults
if "selected_gov_id" not in st.session_state:
    st.session_state["selected_gov_id"] = int(gdf["gov_id"].dropna().iloc[0])

gov_options = gdf.sort_values(display_col)[["gov_id", display_col]].copy()
gov_options["gov_id"] = pd.to_numeric(gov_options["gov_id"], errors="coerce").fillna(-1).astype(int)
gov_id_list = gov_options["gov_id"].tolist()

# Select A
a_choice = st.sidebar.selectbox(
    "Governorate (A)" if lang == "English" else "المحافظة (A)",
    options=gov_id_list,
    format_func=lambda gid: gov_options.loc[gov_options["gov_id"] == gid, display_col].iloc[0],
    index=gov_id_list.index(st.session_state["selected_gov_id"])
    if st.session_state["selected_gov_id"] in gov_id_list
    else 0,
)
st.session_state["selected_gov_id"] = int(a_choice)

# Select B
b_default = next((gid for gid in gov_id_list if gid != st.session_state["selected_gov_id"]), gov_id_list[0])
b_choice = st.sidebar.selectbox(
    "Compare with (B)" if lang == "English" else "قارن مع (B)",
    options=gov_id_list,
    format_func=lambda gid: gov_options.loc[gov_options["gov_id"] == gid, display_col].iloc[0],
    index=gov_id_list.index(b_default),
)
compare_id = int(b_choice)


# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Click a governorate" if lang == "English" else "اضغط على محافظة")

    m = leafmap.Map(center=[26.8, 30.8], zoom=6)
    gjson = gdf.to_json()

    def style_fn(_feature):
        return {"weight": 1, "color": "black", "fillOpacity": 0.4}

    def highlight_fn(_feature):
        return {"weight": 2, "color": "black", "fillOpacity": 0.7}

    tooltip_fields = ["gov_id", "name_en", "name_ar", indicator_col]
    tooltip_aliases = ["ID", "English", "Arabic", indicator_label]
    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True)

    folium.Choropleth(
        geo_data=gjson,
        data=gdf,
        columns=["gov_id", indicator_col],
        key_on="feature.properties.gov_id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="lightgray",
        legend_name=indicator_label,
    ).add_to(m)

    folium.GeoJson(
        data=gjson,
        name="Governorates",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Highlight selected A
    sel = gdf[gdf["gov_id"] == st.session_state["selected_gov_id"]]
    if len(sel) > 0:
        folium.GeoJson(
            data=sel.to_json(),
            name="Selected A",
            style_function=lambda _f: {"weight": 3, "color": "black", "fillOpacity": 0.05},
        ).add_to(m)

    out = st_folium(m, height=600, width=None)

    # Click capture
    clicked = None
    if out and out.get("last_active_drawing") and out["last_active_drawing"].get("properties"):
        clicked = out["last_active_drawing"]["properties"].get("gov_id")

    if clicked is not None:
        st.session_state["selected_gov_id"] = int(clicked)


with col2:
    sel_rows = gdf[gdf["gov_id"] == st.session_state["selected_gov_id"]]
    selected_row = sel_rows.iloc[0] if len(sel_rows) > 0 else gdf.iloc[0]

    st.subheader("Governorate Profile" if lang == "English" else "ملف المحافظة")
    st.markdown(f"### {selected_row['name_en']}")
    st.markdown(f"**{selected_row['name_ar']}**")

    stats = compute_stats(gdf, indicator_col)
    val_num = to_num(selected_row.get(indicator_col))  # ✅ FIXED

    st.divider()
    st.subheader("Quick Insights" if lang == "English" else "ملخص سريع")

    if stats["n"] > 0 and pd.notna(val_num) and stats["ranks"] is not None:
        rank = int(stats["ranks"].loc[selected_row.name])
        n = stats["n"]
        percentile = 100.0 * (n - rank) / (n - 1) if n > 1 else 100.0
        wow = wow_sentence(lang, indicator_col, percentile)

        if lang == "English":
            st.write(f"**Rank:** {rank} / {n}")
            st.write(f"**Percentile:** {percentile:.0f}th")
            st.write(f"**National average:** {stats['mean']:.1f}")
            st.write(f"**National min–max:** {stats['min']:.1f} – {stats['max']:.1f}")
        else:
            st.write(f"**الترتيب:** {rank} / {n}")
            st.write(f"**المئين:** {percentile:.0f}")
            st.write(f"**المتوسط الوطني:** {stats['mean']:.1f}")
            st.write(f"**الحد الأدنى–الأقصى:** {stats['min']:.1f} – {stats['max']:.1f}")

        st.info(wow)
    else:
        st.info("No data for this indicator yet." if lang == "English" else "لا توجد بيانات لهذا المؤشر بعد.")

    # Compare A vs B
    st.divider()
    st.subheader("Compare A vs B" if lang == "English" else "مقارنة A و B")

    row_a_df = gdf[gdf["gov_id"] == st.session_state["selected_gov_id"]]
    row_b_df = gdf[gdf["gov_id"] == compare_id]

    if len(row_a_df) == 0 or len(row_b_df) == 0:
        st.warning("Could not find one of the selected governorates.")
    else:
        row_a = row_a_df.iloc[0]
        row_b = row_b_df.iloc[0]

        name_a = row_a["name_en"] if lang == "English" else row_a["name_ar"]
        name_b = row_b["name_en"] if lang == "English" else row_b["name_ar"]

        val_a = to_num(row_a.get(indicator_col))  # ✅ FIXED
        val_b = to_num(row_b.get(indicator_col))  # ✅ FIXED

        def fmt_val(x):
            if pd.isna(x):
                return "—"
            if indicator_col == "compactness":
                return f"{float(x):.3f}"
            return f"{float(x):,.1f}"

        delta = float(val_b - val_a) if (pd.notna(val_a) and pd.notna(val_b)) else None
        pct = (100.0 * delta / float(val_a)) if (delta is not None and pd.notna(val_a) and float(val_a) != 0) else None

        def fmt_delta(d):
            if d is None:
                return "—"
            sign = "+" if d > 0 else ""
            return f"{sign}{d:.3f}" if indicator_col == "compactness" else f"{sign}{d:,.1f}"

        def fmt_pct(p):
            if p is None:
                return "—"
            sign = "+" if p > 0 else ""
            return f"{sign}{p:.1f}%"

        comp_tbl = pd.DataFrame(
            {
                "Item" if lang == "English" else "البند": [
                    "Governorate" if lang == "English" else "المحافظة",
                    indicator_label,
                    "Δ (B − A)" if lang == "English" else "الفرق (B − A)",
                    "%Δ vs A" if lang == "English" else "نسبة التغير مقابل A",
                ],
                "A": [name_a, fmt_val(val_a), "—", "—"],
                "B": [name_b, fmt_val(val_b), fmt_delta(delta), fmt_pct(pct)],
            }
        )
        st.dataframe(comp_tbl, use_container_width=True)

    # League table
    st.divider()
    st.subheader("League Table" if lang == "English" else "جدول الترتيب")

    top_df, bottom_df = top_bottom_table(gdf, indicator_col, n=5)

    if len(top_df) == 0:
        st.info("Not enough data to build a table." if lang == "English" else "لا توجد بيانات كافية لعرض الجدول.")
    else:
        if lang == "English":
            top_show = top_df[["name_en", indicator_col]].copy()
            top_show.columns = ["Top 5", indicator_label]
            top_show[indicator_label] = top_show[indicator_label].map(lambda x: format_value(indicator_col, x))

            bottom_show = bottom_df[["name_en", indicator_col]].copy()
            bottom_show.columns = ["Bottom 5", indicator_label]
            bottom_show[indicator_label] = bottom_show[indicator_label].map(lambda x: format_value(indicator_col, x))
        else:
            top_show = top_df[["name_ar", indicator_col]].copy()
            top_show.columns = ["أعلى 5", indicator_label]
            top_show[indicator_label] = top_show[indicator_label].map(lambda x: format_value(indicator_col, x))

            bottom_show = bottom_df[["name_ar", indicator_col]].copy()
            bottom_show.columns = ["أدنى 5", indicator_label]
            bottom_show[indicator_label] = bottom_show[indicator_label].map(lambda x: format_value(indicator_col, x))

        st.write(top_show, use_container_width=True)
        st.write(bottom_show, use_container_width=True)
