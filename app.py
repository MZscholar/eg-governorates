from pathlib import Path

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium

# -------------------------------
# Page config (must be first Streamlit call)
# -------------------------------
st.set_page_config(page_title="Egypt Governorates Atlas", layout="wide")

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_TABLES = ROOT / "data" / "tables"

GEO_PATH = DATA_PROCESSED / "governorates.geojson"
POP_PATH = DATA_PROCESSED / "population.csv"
IND_PATH = DATA_TABLES / "indicators.csv"

# -------------------------------
# Sidebar: Data status
# -------------------------------
with st.sidebar.expander("Data status", expanded=False):
    st.write("GeoJSON:", GEO_PATH.name, "✅" if GEO_PATH.exists() else "❌")
    st.write("Population CSV:", POP_PATH.name, "✅" if POP_PATH.exists() else "❌")
    st.write("Indicators CSV:", IND_PATH.name, "✅" if IND_PATH.exists() else "❌")

# -------------------------------
# Small CSS (legend font)
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
# Helpers: safe column dedupe (prevents GeoDataFrame duplicated columns error)
# -------------------------------
def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for j, i in enumerate(dup_idx):
            if j == 0:
                continue
            cols.iloc[i] = f"{dup}__dup{j}"
    df.columns = cols.tolist()
    return df

# -------------------------------
# Loaders (no src/ imports)
# -------------------------------
def load_geo(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "gov_id" not in gdf.columns:
        raise ValueError("GeoJSON must include a 'gov_id' field.")
    gdf["gov_id"] = pd.to_numeric(gdf["gov_id"], errors="coerce").astype("Int64")
    return gdf

def load_population(path: Path) -> pd.DataFrame:
    pop = pd.read_csv(path)
    if "gov_id" not in pop.columns:
        raise ValueError("population.csv must include a 'gov_id' column.")
    pop["gov_id"] = pd.to_numeric(pop["gov_id"], errors="coerce").astype("Int64")
    return pop

def merge_population(gdf: gpd.GeoDataFrame, pop: pd.DataFrame, year: int) -> gpd.GeoDataFrame:
    # Try common column patterns
    candidates = [str(year), f"population_{year}", "population_2023", "population"]
    year_col = next((c for c in candidates if c in pop.columns), None)

    if year_col is None:
        # last resort: first non-gov_id column
        other_cols = [c for c in pop.columns if c != "gov_id"]
        year_col = other_cols[0] if other_cols else None

    if year_col is None:
        gdf["population_2023"] = 0.0
        return gdf

    pop2 = pop[["gov_id", year_col]].copy()
    pop2 = pop2.rename(columns={year_col: "population_2023"})
    pop2["population_2023"] = pd.to_numeric(pop2["population_2023"], errors="coerce").fillna(0)

    out = gdf.merge(pop2, on="gov_id", how="left")
    out["population_2023"] = pd.to_numeric(out["population_2023"], errors="coerce").fillna(0)
    return out

# -------------------------------
# Load data (cached)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(geo_path: str, pop_path: str, ind_path: str, year: int) -> gpd.GeoDataFrame:
    gdf = load_geo(Path(geo_path))

    # Population
    if Path(pop_path).exists():
        pop = load_population(Path(pop_path))
        gdf = merge_population(gdf, pop, year=year)
    else:
        gdf["population_2023"] = 0.0

    # Indicators
    if Path(ind_path).exists():
        indicators = pd.read_csv(ind_path)
        indicators = dedupe_columns(indicators)

        # Normalize gov_id
        indicators["gov_id"] = pd.to_numeric(indicators["gov_id"], errors="coerce").astype("Int64")

        # Avoid duplicate column names after merge
        gdf = dedupe_columns(gdf)

        # Merge
        gdf = gdf.merge(indicators, on="gov_id", how="left")
        gdf = dedupe_columns(gdf)

    # Ensure CRS for web mapping
    try:
        if gdf.crs is not None and str(gdf.crs).lower() != "epsg:4326":
            gdf = gdf.to_crs(epsg=4326)
        elif gdf.crs is None:
            # assume already lat/lon if undefined
            pass
    except Exception:
        pass

    # Ensure numeric type
    gdf["population_2023"] = pd.to_numeric(gdf.get("population_2023", 0), errors="coerce").fillna(0)

    return gdf

DEFAULT_YEAR = 2023  # <-- IMPORTANT: match your indicator label
gdf = load_data(str(GEO_PATH), str(POP_PATH), str(IND_PATH), DEFAULT_YEAR)

# -------------------------------
# Guardrails (only if missing)
# -------------------------------
for col in ["area_km2", "perimeter_km", "compactness"]:
    if col not in gdf.columns:
        gdf[col] = 0.0

# Names fallback
if "name_en" not in gdf.columns:
    gdf["name_en"] = gdf["gov_id"].astype(str)
if "name_ar" not in gdf.columns:
    gdf["name_ar"] = gdf["name_en"].astype(str)

# -------------------------------
# Metric help
# -------------------------------
metric_help = {
    "population_2023": {
        "en": "Residents (from population.csv).",
        "ar": "عدد السكان (من ملف population.csv).",
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

indicator_options = (
    {"Population (2023)": "population_2023", "Area (km²)": "area_km2", "Perimeter (km)": "perimeter_km", "Compactness Index": "compactness"}
    if lang == "English"
    else {"عدد السكان (2023)": "population_2023", "المساحة (كم²)": "area_km2", "المحيط (كم)": "perimeter_km", "مؤشر التماسك": "compactness"}
)

indicator_label = st.sidebar.selectbox("Indicator" if lang == "English" else "المؤشر", list(indicator_options.keys()))
indicator_col = indicator_options[indicator_label]

desc = metric_help.get(indicator_col, {})
st.sidebar.caption(desc.get("en", "") if lang == "English" else desc.get("ar", ""))

# Governorate A / B
gov_options = gdf[[ "gov_id", "name_en", "name_ar" ]].dropna(subset=["gov_id"]).copy()
gov_options["gov_id"] = gov_options["gov_id"].astype(int)

gov_id_list = gov_options["gov_id"].tolist()

a_id = st.sidebar.selectbox(
    "Governorate (A)" if lang == "English" else "المحافظة (A)",
    options=gov_id_list,
    format_func=lambda gid: gov_options.loc[gov_options["gov_id"] == gid, display_col].iloc[0],
    index=0,
)

# B defaults to a different one (safe)
b_candidates = [int(x) for x in gov_id_list if int(x) != int(a_id)]
b_default = b_candidates[0] if b_candidates else int(a_id)

# Safe index (avoid ValueError if not found)
try:
    b_index = [int(x) for x in gov_id_list].index(int(b_default))
except ValueError:
    b_index = 0

b_id = st.sidebar.selectbox(
    "Compare with (B)" if lang == "English" else "قارن مع (B)",
    options=gov_id_list,
    format_func=lambda gid: gov_options.loc[gov_options["gov_id"] == int(gid), display_col].iloc[0],
    index=b_index,
)

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Click a governorate" if lang == "English" else "اضغط على محافظة")

    # IMPORTANT: use folium.Map directly (most stable on Streamlit Cloud)
    m = folium.Map(location=[26.8, 30.8], zoom_start=6, tiles="cartodbpositron")

    gjson = gdf.to_json()

    def style_fn(_feature):
        return {"weight": 1, "color": "black", "fillOpacity": 0.2}

    def highlight_fn(_feature):
        return {"weight": 2, "color": "black", "fillOpacity": 0.6}

    # Choropleth
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

    # Tooltip overlay
    tooltip = folium.GeoJsonTooltip(
        fields=["gov_id", "name_en", "name_ar", indicator_col],
        aliases=["ID", "English", "Arabic", indicator_label],
        sticky=True,
    )

    folium.GeoJson(
        data=gjson,
        name="Governorates",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Render
    st_folium(m, height=650, width=None)

with col2:
    row_a = gdf[gdf["gov_id"].astype(int) == int(a_id)].iloc[0]
    name_a = row_a["name_en"] if lang == "English" else row_a["name_ar"]

    st.subheader("Governorate Profile" if lang == "English" else "ملف المحافظة")
    st.markdown(f"### {row_a['name_en']}")
    st.markdown(f"**{row_a['name_ar']}**")

    st.divider()
    st.markdown(f"**{indicator_label}**")

    val = pd.to_numeric(row_a.get(indicator_col), errors="coerce")
    if pd.isna(val):
        st.write("—")
    else:
        if indicator_col == "compactness":
            st.write(f"{float(val):.3f}")
        else:
            st.write(f"{float(val):,.1f}")

    st.divider()
    st.subheader("Compare A vs B" if lang == "English" else "مقارنة A و B")

    row_b = gdf[gdf["gov_id"].astype(int) == int(b_id)].iloc[0]
    name_b = row_b["name_en"] if lang == "English" else row_b["name_ar"]

    val_a = pd.to_numeric(row_a.get(indicator_col), errors="coerce")
    val_b = pd.to_numeric(row_b.get(indicator_col), errors="coerce")

    def fmt(x):
        if pd.isna(x):
            return "—"
        if indicator_col == "compactness":
            return f"{float(x):.3f}"
        return f"{float(x):,.1f}"

    delta = (val_b - val_a) if (pd.notna(val_a) and pd.notna(val_b)) else None

    st.write(f"**A:** {name_a} → {fmt(val_a)}")
    st.write(f"**B:** {name_b} → {fmt(val_b)}")
    st.write(f"**Δ (B − A):** {fmt(delta) if delta is not None else '—'}")

