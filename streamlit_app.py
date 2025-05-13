from collections import defaultdict
from pathlib import Path
import streamlit as st
import altair as alt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import time
from pytrends.exceptions import TooManyRequestsError

# Function to fetch Google Trends with retry logic and exponential backoff
@st.cache_data(ttl=86400)
def fetch_google_trends(keyword, timeframe='today 3-m', retries=3, sleep_base=5):
    pytrends = TrendReq(hl='en-US', tz=360)
    for attempt in range(retries):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            data = pytrends.interest_over_time()
            if data.empty:
                return pd.DataFrame(columns=['Date', 'Google_Trend'])
            return data.reset_index()[['date', keyword]].rename(columns={'date': 'Date', keyword: 'Google_Trend'})
        except TooManyRequestsError as e:
            time.sleep(sleep_base * (attempt + 1))
        except Exception as e:
            st.warning(f"⚠️ Google Trends fetch failed: {e}")
            break
    st.warning(f"⚠️ Google Trends unavailable after {retries} attempts.")
    return pd.DataFrame(columns=['Date', 'Google_Trend'])

# Set up enhanced page config and theme
st.set_page_config(page_title="MindGames Dashboard", layout="wide", page_icon="🧠")

# Custom CSS for modern UI
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
            margin: auto;
        }
        h1, h2, h3 {
            color: #1E293B;
            font-family: 'Segoe UI', sans-serif;
        }
        .metric-box {
            background: linear-gradient(to right, #6366F1, #3B82F6);
            color: white;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        }
        .section {
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #E5E7EB;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align:center">
        <h1>🧠 MindGames Inventory Intelligence</h1>
        <p style="color:gray">Monitor, Forecast & Optimize Inventory Across All Locations</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar filters
inventory_file = st.sidebar.file_uploader("Upload inventory snapshot (CSV)", type="csv", key="inv")
sales_file = st.sidebar.file_uploader("Upload sales history (item_name, Date, Units_Sold, location)", type="csv", key="sales_history")
st.sidebar.header("📂 Filters")
sku_file = st.sidebar.file_uploader("Upload SKU list (CSV or TXT)", type=["csv", "txt"])
sku_list = []
if sku_file:
    try:
        sku_df = pd.read_csv(sku_file, header=None)
        sku_list = sku_df[0].astype(str).tolist()
    except Exception as e:
        st.error(f"Error reading SKU file: {e}")

trend_keyword = st.sidebar.text_input("🔍 Enter keyword for Google Trends", value="Magic Cards")

# Load inventory data directly from uploaded CSV or fallback
if inventory_file:
    df = pd.read_csv(inventory_file)
else:
    st.warning("⚠️ Please upload an inventory snapshot CSV file.")
    st.stop()

categories = df["category"].dropna().unique().tolist()
suppliers = df["supplier"].dropna().unique().tolist()
locations = df["location"].dropna().unique().tolist()

selected_categories = st.sidebar.multiselect("Category", categories)
selected_suppliers = st.sidebar.multiselect("Supplier", suppliers)
selected_locations = st.sidebar.multiselect("Location", locations)

region_scope = st.sidebar.radio("Select Region Scope", ["All", "CA", "US"], index=0)

@st.cache_data(ttl=300)
def get_filtered_data(df, categories, suppliers, locations, sku_list, region_scope):
    if region_scope == "CA":
        df = df[df["location"].str.contains("CA") | (df["location"] == "Main Warehouse")]
    elif region_scope == "US":
        df = df[df["location"].str.contains("US")]

    supplier_filter = df["supplier"].isin(suppliers) if suppliers else pd.Series([True] * len(df))
    category_filter = df["category"].isin(categories) if categories else pd.Series([True] * len(df))
    location_filter = df["location"].isin(locations) if locations else pd.Series([True] * len(df))
    filtered = df[supplier_filter & category_filter & location_filter]

    if sku_list:
        filtered = filtered[filtered['item_name'].astype(str).isin(sku_list)]
    return filtered

filtered_df = get_filtered_data(df, selected_categories, selected_suppliers, selected_locations, sku_list, region_scope)

# Compute KPIs
filtered_df["margin_%"] = ((filtered_df["price"] - filtered_df["cost_price"]) / filtered_df["price"] * 100).round(2)
filtered_df["stock_value"] = (filtered_df["cost_price"] * filtered_df["units_left"]).round(2)
filtered_df["inventory_turnover"] = (filtered_df["units_sold"] / (filtered_df["units_sold"] + filtered_df["units_left"] + 1e-9)).round(2)

# KPI layout
st.markdown('<div class="section"><h2>📊 Key Metrics</h2></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-box'><h3>${filtered_df['stock_value'].sum():,.2f}</h3><p>Total Stock Value</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box'><h3>{filtered_df['margin_%'].mean():.2f}%</h3><p>Avg Margin</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-box'><h3>{filtered_df['inventory_turnover'].mean():.2f}</h3><p>Inventory Turnover</p></div>", unsafe_allow_html=True)

# Inventory Table
st.markdown('<div class="section"><h2>📋 Inventory Table</h2></div>', unsafe_allow_html=True)
page_size = 50
max_pages = max((len(filtered_df) - 1) // page_size + 1, 1)
page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
paginated_df = filtered_df.iloc[(page - 1) * page_size: page * page_size]
st.dataframe(paginated_df, use_container_width=True)

# Low stock
st.markdown('<div class="section"><h2>🧯 Low Stock Alerts</h2></div>', unsafe_allow_html=True)
low_stock = filtered_df[filtered_df["units_left"] < filtered_df["reorder_point"]]
if not low_stock.empty:
    st.warning("Some items are below reorder threshold.")
    st.dataframe(low_stock[["item_name", "location", "units_left", "reorder_point"]])

# Charts
st.markdown('<div class="section"><h2>🔥 Top Sellers</h2></div>', unsafe_allow_html=True)
st.altair_chart(
    alt.Chart(filtered_df).mark_bar(orient="horizontal").encode(
        x="units_sold", y=alt.Y("item_name").sort("-x"), tooltip=["units_sold", "item_name"]
    ), use_container_width=True
)

# Category/Supplier
st.markdown('<div class="section"><h2>📦 Category Summary</h2></div>', unsafe_allow_html=True)
st.dataframe(
    filtered_df.groupby("category").agg(
        total_stock_value=("stock_value", "sum"),
        avg_margin=("margin_%", "mean"),
        turnover=("inventory_turnover", "mean")
    ).round(2).reset_index()
)

st.markdown('<div class="section"><h2>🤝 Supplier Summary</h2></div>', unsafe_allow_html=True)
st.dataframe(
    filtered_df.groupby("supplier").agg(
        total_stock_value=("stock_value", "sum"),
        avg_margin=("margin_%", "mean"),
        turnover=("inventory_turnover", "mean")
    ).round(2).reset_index()
)

# Forecasting & Lifecycle
# if sales_file:
#     try:
#         sales_data = pd.read_csv(sales_file, parse_dates=["Date"])
#         sales_data = sales_data.sort_values(["item_name", "Date"])

#         trend_data = fetch_google_trends(trend_keyword)
#         if not trend_data.empty:
#             sales_data = pd.merge(sales_data, trend_data, on="Date", how="left")

#         st.markdown('<div class="section"><h2>🧠 Product Lifecycle Classification</h2></div>', unsafe_allow_html=True)
#         last_date = sales_data["Date"].max()
#         sales_by_item = sales_data.groupby("item_name").agg(
#             last_sale=("Date", "max"),
#             total_sales=("Units_Sold", "sum"),
#             days_active=("Date", lambda x: (x.max() - x.min()).days + 1),
#             unique_weeks=("Date", lambda x: len(set(x.dt.to_period("W"))))
#         ).reset_index()

#         def classify(row):
#             days_since_last = (last_date - row["last_sale"]).days
#             if days_since_last > 180:
#                 return "❌ Outdated (No Sales 6+ Months)"
#             elif row["unique_weeks"] >= 20:
#                 return "✅ Evergreen (Always in Stock)"
#             elif row["unique_weeks"] > 5:
#                 return "🎯 Seasonal"
#             else:
#                 return "📉 Low Activity"

#         sales_by_item["Lifecycle Category"] = sales_by_item.apply(classify, axis=1)
#         st.dataframe(sales_by_item[["item_name", "last_sale", "total_sales", "Lifecycle Category"]])

#         st.markdown('<div class="section"><h2>📈 Forecasting & Demand Planning</h2></div>', unsafe_allow_html=True)
#         today = pd.Timestamp.today()
#         summary_rows = []

#         for sku in sales_data["item_name"].unique():
#             for location in sales_data["location"].unique():
#                 df_item = sales_data[
#                     (sales_data["item_name"] == sku) &
#                     (sales_data["location"] == location)
#                 ].set_index("Date")["Units_Sold"].resample("D").sum().fillna(0)

#                 row = {"Item": sku, "Location": location}
#                 row["Last 30 Days"] = df_item[-30:].sum()
#                 row["Last 60 Days"] = df_item[-60:].sum()
#                 row["Last 90 Days"] = df_item[-90:].sum()
#                 row["YTD"] = df_item[df_item.index >= pd.Timestamp(today.year, 1, 1)].sum()

#                 last_year = today.year - 1
#                 for days in [30, 60, 90]:
#                     ly_start = today.replace(year=last_year)
#                     ly_end = ly_start + pd.Timedelta(days=days)
#                     mask = (df_item.index >= ly_start) & (df_item.index < ly_end)
#                     row[f"Next {days} LY"] = df_item.loc[mask].sum()

#                 if len(df_item) >= 60 and sku in sales_by_item[sales_by_item['Lifecycle Category'] != '❌ Outdated (No Sales 6+ Months)']["item_name"].values:
#                     model = ExponentialSmoothing(df_item, trend="add", seasonal=None).fit()
#                     forecast = model.forecast(30)
#                     forecast_total = forecast.sum()
#                     row["Forecast Next 30"] = forecast_total
#                     row["Lifecycle"] = sales_by_item[sales_by_item["item_name"] == sku]["Lifecycle Category"].values[0]

#                     current_stock = filtered_df[(filtered_df["item_name"] == sku) & (filtered_df["location"] == location)]
#                     units_left = int(current_stock["units_left"].values[0]) if not current_stock.empty else 0
#                     reorder_qty = max(int(forecast_total) - units_left, 0)
#                     row["Units Left"] = units_left
#                     row["Suggested Reorder"] = reorder_qty

#                 summary_rows.append(row)

#         demand_df = pd.DataFrame(summary_rows)
#         st.dataframe(demand_df, use_container_width=True)
#         st.download_button("⬇ Download Demand Plan", demand_df.to_csv(index=False), "demand_plan.csv")

#     except Exception as e:
#         st.error(f"Error processing sales file: {e}")

st.caption("Made for MindGames — Powered by Streamlit | All metrics reflect applied filters.")
