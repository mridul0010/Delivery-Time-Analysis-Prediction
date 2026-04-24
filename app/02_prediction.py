import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Delivery Time Analysis",
    page_icon="🚚",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ======================== FEATURE ENGINEERING ========================

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance using haversine formula."""
    R = 6371  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def engineer_features(df):
    """Apply feature engineering from EDA notebook."""
    df = df.copy()
    
    # DateTime conversions
    df['Order_Datetime'] = pd.to_datetime(df['Order_Datetime'])
    df['Pickup_Datetime'] = pd.to_datetime(df['Pickup_Datetime'])
    
    # Distance calculation
    distance_km = haversine(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"]
    )
    df["distance_km"] = distance_km
    
    # Direction calculation
    lat_diff = df["Delivery_location_latitude"] - df["Restaurant_latitude"]
    lon_diff = df["Delivery_location_longitude"] - df["Restaurant_longitude"]
    direction = np.arctan2(lat_diff, lon_diff)
    df["direction_rad"] = direction
    
    # Distance groups
    distance_group = pd.cut(
        df['distance_km'],
        bins=[0, 5, 10, 25],
        labels=['Short Distance', 'Median Distance', 'Long Distance']
    )
    df["Distance_Group"] = distance_group
    
    # Speed calculation
    df['speed_kmph'] = df['distance_km'] / (df['Time_taken (min)'] / 60)
    df = df[(df['speed_kmph'] <= 60) & (df['speed_kmph'] >= 3)]
    
    # Time of day
    df['Order_hour'] = df['Order_Datetime'].dt.hour
    df['Order_day'] = df['Order_Datetime'].dt.day_name()
    df['isWeekend'] = df['Order_day'].isin(["Saturday", "Sunday"]).astype(int)
    
    df['Time_Of_Day'] = pd.cut(
        df['Order_hour'],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        include_lowest=True
    )
    
    # Remove Semi-Urban (unrealistic patterns)
    df = df[df['City_Type'] != "Semi-Urban"]
    
    # Traffic-distance interaction
    df['distance_traffic'] = df['distance_km'] * df['Road_traffic_density'].map(
        {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    )
    
    return df


# ======================== DATA LOADING & PREPROCESSING ========================

@st.cache_data
def load_and_preprocess():
    """Load and preprocess delivery dataset."""
    df = pd.read_csv("data/Cleaned Delivery Dataset.csv")
    df = engineer_features(df)
    return df


df = load_and_preprocess()

# ======================== SIDEBAR FILTERS ========================
st.sidebar.header("🔎 Filters")

cities = sorted(df["City"].unique())
zones = sorted(df["Zone"].unique())
traffic_levels = sorted(df["Road_traffic_density"].unique())
weather_types = sorted(df["Weather_conditions"].unique())
time_periods = ["Night", "Morning", "Afternoon", "Evening"]
vehicle_types = sorted(df["Type_of_vehicle"].unique())
city_types = sorted(df["City_Type"].unique())

sel_cities = st.sidebar.multiselect("City", cities)
sel_zones = st.sidebar.multiselect("Zone", zones)
sel_traffic = st.sidebar.multiselect("Traffic Density", traffic_levels)
sel_weather = st.sidebar.multiselect("Weather", weather_types)
sel_time = st.sidebar.multiselect("Time of Day", time_periods)
sel_vehicle = st.sidebar.multiselect("Vehicle Type", vehicle_types)
sel_city_type = st.sidebar.multiselect("City Type", city_types)

distance_range = st.sidebar.slider(
    "Distance Range (km)",
    float(df["distance_km"].min()),
    float(df["distance_km"].max()),
    (float(df["distance_km"].min()), float(df["distance_km"].max())),
)

fdf = df.copy()
if sel_cities:
    fdf = fdf[fdf["City"].isin(sel_cities)]
if sel_zones:
    fdf = fdf[fdf["Zone"].isin(sel_zones)]
if sel_traffic:
    fdf = fdf[fdf["Road_traffic_density"].isin(sel_traffic)]
if sel_weather:
    fdf = fdf[fdf["Weather_conditions"].isin(sel_weather)]
if sel_time:
    fdf = fdf[fdf["Time_Of_Day"].isin(sel_time)]
if sel_vehicle:
    fdf = fdf[fdf["Type_of_vehicle"].isin(sel_vehicle)]
if sel_city_type:
    fdf = fdf[fdf["City_Type"].isin(sel_city_type)]

fdf = fdf[(fdf["distance_km"] >= distance_range[0]) & (fdf["distance_km"] <= distance_range[1])]

n = len(fdf)

# ======================== HEADER ========================
st.title("🚚 Delivery Time Prediction — Analytics Dashboard")
st.caption("Interactive dashboard analyzing delivery performance factors")

# ======================== KPI CARDS ========================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Deliveries", f"{n:,}")
k2.metric("Avg Delivery Time", f"{fdf['Time_taken (min)'].mean():.1f} min" if n else "—")
k3.metric("Avg Speed", f"{fdf['speed_kmph'].mean():.1f} km/h" if n else "—")
k4.metric("Avg Distance", f"{fdf['distance_km'].mean():.1f} km" if n else "—")
k5.metric("Cities Covered", fdf["City"].nunique())

if n == 0:
    st.warning("No data matches the current filters. Adjust filters in the sidebar.")
    st.stop()

# ======================== TAB LAYOUT ========================
tab_overview, tab_traffic, tab_distance, tab_time, tab_partners, tab_orders, tab_data, tab_insights = st.tabs(
    ["📊 Overview", "🚦 Traffic Impact", "📍 Distance Analysis", "⏰ Time Analysis", 
     "🚴 Delivery Partners", "📦 Order Analysis", "📋 Data", "🎯 Final Insights"]
)

# ================================================================
# TAB 1 — OVERVIEW
# ================================================================
with tab_overview:
    st.subheader("Delivery Time Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="Time_taken (min)", nbins=40, marginal="box",
                          color_discrete_sequence=["#667eea"])
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(fdf, y="Time_taken (min)", color="City_Type",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📊 Insight:** The delivery time distribution shows how deliveries are spread across different durations. 
    Most deliveries cluster around the average, with Urban areas showing faster and more consistent delivery times compared to Metropolitan areas.
    """)

    st.divider()

    st.subheader("Speed Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="speed_kmph", nbins=40, marginal="box",
                          color_discrete_sequence=["#764ba2"])
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(fdf, x="Type_of_vehicle", y="speed_kmph",
                    color="Type_of_vehicle", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **⚡ Insight:** Speed varies by vehicle type. Two-wheelers typically maintain higher speeds due to their maneuverability, 
    while three-wheelers and motorcycles have different speed characteristics. Higher speed doesn't always mean faster delivery times 
    due to route planning efficiency.
    """)

    st.divider()

    st.subheader("Deliveries by City")
    city_counts = fdf["City"].value_counts().head(10).reset_index()
    city_counts.columns = ["City", "Count"]
    fig = px.bar(city_counts, x="City", y="Count", color="Count",
                color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🏙️ Insight:** Some cities generate significantly more orders than others. The major revenue-generating cities include 
    Jaipur, Surat, Bangalore, and others with high delivery volumes, indicating market concentration and demand patterns.
    """)

# ================================================================
# TAB 2 — TRAFFIC IMPACT
# ================================================================
with tab_traffic:
    st.subheader("🚦 Traffic Density Impact on Delivery Time")
    
    c1, c2 = st.columns(2)
    with c1:
        traffic_time = fdf.groupby("Road_traffic_density")["Time_taken (min)"].mean().reset_index()
        traffic_time = traffic_time.sort_values("Time_taken (min)")
        fig = px.bar(traffic_time, x="Road_traffic_density", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.box(fdf, x="Road_traffic_density", y="Time_taken (min)",
                    color="Road_traffic_density",
                    color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **🚦 CRITICAL INSIGHT - Traffic is the PRIMARY FACTOR:**
    - 🟣 Low Traffic: Fastest deliveries (~15-25 min)
    - 🟢 Medium Traffic: Moderate delays (~20-30 min)
    - 🔴 High Traffic: Significant delays (~25-35 min)
    - 🚨 Jam Conditions: Severe delays (~30-45+ min)
    
    **Traffic congestion is the most critical factor affecting delivery efficiency.**
    """)

    st.divider()

    st.subheader("🌦️ Traffic × Weather Interaction")
    traffic_weather = fdf.groupby(["Road_traffic_density", "Weather_conditions"])["Time_taken (min)"].mean().reset_index()
    fig = px.bar(traffic_weather, x="Road_traffic_density", y="Time_taken (min)", 
                color="Weather_conditions", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🔥 Combined Effect:** Weather and traffic together amplify delays significantly.
    - **Best Case:** Sunny + Low traffic → ~10-20 min
    - **Worst Case:** Fog/Cloudy + Jam → ~40-55 min
    """)

    st.divider()

    st.subheader("Distance vs Delivery Time (colored by Traffic)")
    fig = px.scatter(fdf, x="distance_km", y="Time_taken (min)",
                    color="Road_traffic_density", trendline="ols",
                    hover_data=["City", "Order_hour"],
                    color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📊 Insight:** The scatter plot reveals that traffic density is a stronger predictor than distance alone.
    At the same distance, deliveries in jam conditions take significantly longer than those in low traffic.
    """)

    st.divider()

    st.subheader("🔥 Traffic Impact Heatmap")
    traffic_city = pd.pivot_table(fdf, values="Time_taken (min)",
                                  index="City", columns="Road_traffic_density", 
                                  aggfunc="mean").fillna(0)
    fig = px.imshow(traffic_city, color_continuous_scale="RdYlGn_r", aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🗺️ Insight:** Different cities experience different traffic impacts. Metropolitan and urban areas show stronger 
    correlations between traffic conditions and delivery time, indicating traffic as a city-specific operational challenge.
    """)

# ================================================================
# TAB 3 — DISTANCE ANALYSIS
# ================================================================
with tab_distance:
    st.subheader("📍 Distance Impact Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        dist_group = fdf.groupby("Distance_Group", observed=True)["Time_taken (min)"].mean().reset_index()
        fig = px.bar(dist_group, x="Distance_Group", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.box(fdf, x="Distance_Group", y="Time_taken (min)",
                    color="Distance_Group", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📏 Insight:** Distance strongly correlates with delivery time:
    - Short Distance (0-5 km): ~15-25 min
    - Median Distance (5-10 km): ~20-30 min
    - Long Distance (10-25 km): ~25-40 min
    
    As distance increases, delivery time increases proportionally.
    """)

    st.divider()

    st.subheader("Distance vs Delivery Time (Primary Driver)")
    fig = px.scatter(fdf, x="distance_km", y="Time_taken (min)",
                    color="City_Type", trendline="ols",
                    hover_data=["City", "Road_traffic_density"],
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **✅ POSITIVE RELATIONSHIP:** The upward-sloping trend line confirms:
    **As distance increases → delivery time increases**
    
    This is one of the core drivers of delivery efficiency.
    """)

    st.divider()

    st.subheader("Distance vs Speed Relationship")
    fig = px.scatter(fdf, x="distance_km", y="speed_kmph",
                    color="Road_traffic_density", trendline="ols",
                    color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **💨 Insight:** Speed patterns change with distance and traffic:
    - Longer distances in low traffic → higher speeds
    - Longer distances in jam → much lower speeds (congestion effect)
    """)

    st.divider()

    st.subheader("🔥 Distance × Traffic Interaction Matrix")
    dist_traffic = pd.pivot_table(fdf, values="Time_taken (min)",
                                  index="Distance_Group", columns="Road_traffic_density",
                                  aggfunc="mean").fillna(0)
    fig = px.imshow(dist_traffic, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **🔥 CRITICAL INTERACTION:** Distance matters MORE when traffic is HIGH.
    - Long Distance + Low Traffic: ~25-30 min
    - Long Distance + Jam: ~35-45+ min
    
    This shows that traffic impact is amplified over longer distances.
    """)

# ================================================================
# TAB 4 — TIME ANALYSIS
# ================================================================
with tab_time:
    st.subheader("⏰ Time of Day Impact")
    
    c1, c2 = st.columns(2)
    with c1:
        time_delivery = fdf.groupby("Time_Of_Day", observed=True)["Time_taken (min)"].mean().reset_index()
        fig = px.bar(time_delivery, x="Time_Of_Day", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Sunset")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.box(fdf, x="Time_Of_Day", y="Time_taken (min)",
                    color="Time_Of_Day", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **⏰ TIME OF DAY EFFECT:**
    - 🌅 Morning (Best): ~21 min - Low traffic, lower order volume
    - 🌙 Night (Good): ~24 min - Low traffic outweighs low staffing
    - 🌤 Afternoon: ~26 min - Moderate delays, traffic buildup
    - 🌆 Evening (Worst): ~28+ min - Peak traffic & peak order volume
    
    **"Delivery performance deteriorates during evening hours due to peak demand and congestion."**
    """)

    st.divider()

    st.subheader("📊 Hourly Order Distribution")
    hourly_orders = fdf["Order_hour"].value_counts().sort_index().reset_index()
    hourly_orders.columns = ["Hour", "Count"]
    fig = px.bar(hourly_orders, x="Hour", y="Count", color="Count",
                color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🕐 Insight:** Order patterns show clear peaks:
    - Peak: 18-22 hours (Dinner time) - highest demand
    - Secondary: 8-11 hours (Breakfast/Lunch)
    - Low: 12-16 hours (Afternoon dip)
    """)

    st.divider()

    st.subheader("🔥 Time of Day × Traffic Interaction (MOST IMPORTANT)")
    time_traffic = fdf.groupby(["Time_Of_Day", "Road_traffic_density"], observed=True)["Time_taken (min)"].mean().reset_index()
    fig = px.bar(time_traffic, x="Time_Of_Day", y="Time_taken (min)",
                color="Road_traffic_density", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)

    st.error("""
    **🔥 MOST IMPORTANT INTERACTION EFFECT:**
    - **Morning:** Traffic impact is LOW (even jam doesn't add much)
    - **Evening:** Traffic impact is VERY HIGH (jam conditions cause maximum delays)
    
    **"The effect of traffic is amplified during peak hours, making time-of-day a critical contextual factor."**
    """)

    st.divider()

    st.subheader("Day-wise Performance")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_delivery = fdf.groupby("Order_day")["Time_taken (min)"].mean().reindex(day_order).reset_index()
    day_delivery.columns = ["Day", "Avg Delivery Time"]
    fig = px.bar(day_delivery, x="Day", y="Avg Delivery Time",
                color="Avg Delivery Time", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📅 Day-wise Trends:**
    - ❌ Friday: WORST performance (~28+ min) - End-of-week rush, high demand
    - ✅ Tuesday/Thursday: BEST performance (~26 min) - Stable traffic, lower demand
    """)

    st.divider()

    st.subheader("⏰ Time × Distance Matrix")
    time_dist = pd.pivot_table(fdf, values="Time_taken (min)",
                               index="Time_Of_Day", columns="Distance_Group",
                               aggfunc="mean").fillna(0)
    fig = px.imshow(time_dist, text_auto=".1f", color_continuous_scale="Turbo", aspect="auto")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **⏰📏 Insight:** Time of day affects different distance groups differently.
    Long distances during evening show the most extreme delays.
    """)

# ================================================================
# TAB 5 — DELIVERY PARTNERS
# ================================================================
with tab_partners:
    st.subheader("🚴 Delivery Partner Rating Impact")
    
    c1, c2 = st.columns(2)
    with c1:
        rating_time = fdf.groupby("Delivery_Person_Rating_Group")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(rating_time, x="Delivery_Person_Rating_Group", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        rating_speed = fdf.groupby("Delivery_Person_Rating_Group")["speed_kmph"].mean().reset_index()
        fig = px.bar(rating_speed, x="Delivery_Person_Rating_Group", y="speed_kmph",
                    color="speed_kmph", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **🚴 DELIVERY PARTNER INSIGHTS:**
    - **High-rated riders:** ~9-10 minutes faster delivery times
    - **Not necessarily the highest speed** - Efficiency > Raw Speed
    
    **"Efficiency and decision-making outperform raw speed."**
    
    High-rated riders excel at:
    - ✔️ Better route selection
    - ✔️ Better time management
    - ✔️ Less waiting time
    - ✔️ Smarter navigation
    """)

    st.divider()

    st.subheader("🧑‍🍳 Age Group Performance")
    
    c1, c2 = st.columns(2)
    with c1:
        age_delivery = fdf.groupby("Age_Group")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(age_delivery, x="Age_Group", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        age_speed = fdf.groupby("Age_Group")["speed_kmph"].mean().reset_index()
        fig = px.bar(age_speed, x="Age_Group", y="speed_kmph",
                    color="speed_kmph", color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **👦 Age vs Performance Paradox:**
    - Younger riders → Higher speed (kmph)
    - NOT always best delivery time
    - Experience contributes more to efficiency than speed alone
    
    **"Delivery experience and judgment matter more than raw driving speed."**
    """)

    st.divider()

    st.subheader("Rating × Age Group Interaction")
    rating_age = fdf.groupby(["Age_Group", "Delivery_Person_Rating_Group"])["Time_taken (min)"].mean().reset_index()
    fig = px.bar(rating_age, x="Age_Group", y="Time_taken (min)",
                color="Delivery_Person_Rating_Group", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **⭐ Insight:** Adult delivery agents (25-35) with high ratings show the best performance.
    Rating becomes increasingly important for older age groups where experience shines through.
    """)

    st.divider()

    st.subheader("🔍 Efficiency vs Speed Trade-off")
    st.markdown("""
    ### **Key Insight:** Efficiency ≠ Speed
    
    **High-rated riders characteristics:**
    - Slightly slower speed
    - Fastest overall delivery times
    - Better route planning
    - Smarter decision-making
    
    **Low-rated riders characteristics:**
    - Fastest speed on paper
    - BUT slower deliveries in reality
    - Poor route choices
    - Delays at pickup points
    - Inefficient handling
    """)

# ================================================================
# TAB 6 — ORDER ANALYSIS
# ================================================================
with tab_orders:
    st.subheader("📦 Multiple Deliveries Impact")
    
    c1, c2 = st.columns(2)
    with c1:
        multi_delivery = fdf.groupby("multiple_deliveries")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(multi_delivery, x="multiple_deliveries", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        multi_counts = fdf["multiple_deliveries"].value_counts().sort_index().reset_index()
        multi_counts.columns = ["Deliveries", "Count"]
        fig = px.pie(multi_counts, names="Deliveries", values="Count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.error("""
    **📦 MULTIPLE DELIVERIES - STRONG IMPACT:**
    - 0 deliveries: ~23 min
    - 2 deliveries: ~40 min (+17 min)
    - 3 deliveries: ~47 min (+24 min)
    
    **Non-linear relationship: More deliveries = exponentially higher delays**
    
    **"Higher delivery load increases route complexity and delays."**
    """)

    st.divider()

    st.subheader("🚗 Weather Conditions Impact")
    
    c1, c2 = st.columns(2)
    with c1:
        weather_delivery = fdf.groupby("Weather_conditions")["Time_taken (min)"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(weather_delivery, x="Weather_conditions", y="Time_taken (min)",
                    color="Time_taken (min)", color_continuous_scale="Oranges")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.box(fdf, x="Weather_conditions", y="Time_taken (min)",
                    color="Weather_conditions", color_discrete_sequence=px.colors.qualitative.Light24)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🌦️ WEATHER IMPACT RANKING:**
    - ☀️ Sunny → FASTEST (~20-22 min)
    - ⛈🌪🌬 Stormy/Sandstorms/Windy → Moderate (~25-27 min)
    - 🌫☁️ Fog/Cloudy → SLOWEST (~27-29 min)
    
    **Interesting:** Fog/Cloudy causes more delays than storms!
    - Visibility issues > Rain effects on delivery
    - Riders slow down more in fog than heavy rain
    
    **"Adverse weather moderately increases delivery time."**
    """)

    st.divider()

    st.subheader("📊 Order Type Impact")
    order_type = fdf.groupby("Type_of_order")["Time_taken (min)"].mean().reset_index()
    fig = px.bar(order_type, x="Type_of_order", y="Time_taken (min)",
                color="Time_taken (min)", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📋 Insight:** Order type has NO significant effect on delivery time.
    Logistics factors dominate, not product characteristics.
    """)

    st.divider()

    st.subheader("🚗 Vehicle Type Performance")
    vehicle_delivery = fdf.groupby("Type_of_vehicle")["Time_taken (min)"].mean().reset_index()
    fig = px.bar(vehicle_delivery, x="Type_of_vehicle", y="Time_taken (min)",
                color="Time_taken (min)", color_continuous_scale="Greens")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🚙 Insight:** Different vehicle types show varying performance.
    Two-wheelers typically provide faster delivery times due to maneuverability.
    """)

# ================================================================
# TAB 7 — DATA EXPLORER
# ================================================================
with tab_data:
    st.subheader("📋 Filtered Dataset")
    
    # Select columns to display
    display_cols = st.multiselect(
        "Select columns to display",
        fdf.columns.tolist(),
        default=['City', 'distance_km', 'Time_taken (min)', 'speed_kmph', 
                'Road_traffic_density', 'Weather_conditions', 'Time_Of_Day']
    )
    
    if display_cols:
        st.dataframe(fdf[display_cols], use_container_width=True, height=500)
    else:
        st.dataframe(fdf, use_container_width=True, height=500)
    
    # Download button
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered CSV", data=csv,
                       file_name="filtered_delivery_data.csv", mime="text/csv")

# ================================================================
# TAB 8 — FINAL INSIGHTS
# ================================================================
with tab_insights:
    st.subheader("🎯 Final Insights Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚦 Traffic Impact", f"+{(fdf[fdf['Road_traffic_density']=='Jam']['Time_taken (min)'].mean() - fdf[fdf['Road_traffic_density']=='Low']['Time_taken (min)'].mean()):.1f} min", 
                 "Jam vs Low")
    with col2:
        st.metric("⏰ Time of Day", f"+{(fdf[fdf['Time_Of_Day']=='Evening']['Time_taken (min)'].mean() - fdf[fdf['Time_Of_Day']=='Morning']['Time_taken (min)'].mean()):.1f} min",
                 "Evening vs Morning")
    with col3:
        st.metric("📦 Multiple Deliveries", f"+{(fdf[fdf['multiple_deliveries']==3]['Time_taken (min)'].mean() - fdf[fdf['multiple_deliveries']==0]['Time_taken (min)'].mean()):.1f} min",
                 "3 vs 0 deliveries")

    st.divider()

    st.markdown("""
    # 📊 Final Insights Summary

    ---

    ## **🚀 1. Core Drivers of Delivery Time**
    Delivery time is primarily driven by:
    - **Traffic conditions** (PRIMARY)
    - **Distance** (CORE)
    - **Operational load** (SECONDARY)

    👉 **Traffic is the most dominant factor**

    ---

    ## **🚦 2. Traffic is the Strongest Factor**
    - High traffic significantly increases delivery time  
    - Jam conditions cause the **highest delays**  

    > *"Traffic congestion is the most critical factor affecting delivery efficiency."*

    **Impact:** Jam vs Low Traffic = +8-15 minutes

    ---

    ## **⏰ 3. Time of Day Effect**
    - 🌅 Morning → fastest deliveries (~21 min)  
    - 🌆 Evening → slowest deliveries (~28+ min)  

    > *"Delivery performance deteriorates during evening hours due to peak demand and congestion."*

    **Pattern:** Morning < Night << Afternoon < Evening

    ---

    ## **🔥 4. Time × Traffic Interaction (MOST IMPORTANT)**
    - Morning: traffic impact is **low**  
    - Evening: traffic impact is **very high**  

    > *"The effect of traffic is amplified during peak hours, making time-of-day a critical contextual factor."*

    **Best Case:** Morning + Low Traffic ≈ 19-21 min
    **Worst Case:** Evening + Jam ≈ 31-45+ min

    ---

    ## **📦 5. Multiple Deliveries (Operational Load)**
    - 0 deliveries → ~23 min  
    - 2 deliveries → ~40 min  
    - 3 deliveries → ~47 min  

    👉 Delivery time increases **non-linearly**

    > *"Higher delivery load increases route complexity and delays."*

    ---

    ## **🚴 6. Delivery Partner Insights**
    - High-rated riders → **fastest deliveries** (~9-10 min faster)
    - Not necessarily the highest speed  

    > *"Efficiency and decision-making outperform raw speed."*

    **Key:** Efficiency > Speed
    - High-rated: Better route planning, time management, navigation
    - Low-rated: Fast but inefficient, poor routes, pickup delays

    ---

    ## **👦 7. Age vs Performance**
    - Younger riders → higher speed  
    - Not always best delivery time  

    > *"Experience contributes more to efficiency than speed alone."*

    **Best Performers:** Adult riders (25-35) with high ratings

    ---

    ## **🌦️ 8. Weather Impact**
    - Fog/Cloudy → higher delays (~27-29 min)
    - Sunny → faster deliveries (~20-22 min)

    > *"Adverse weather moderately increases delivery time."*

    **Ranking:** Sunny < Stormy/Windy < Fog/Cloudy

    ---

    ## **📅 9. Day-wise Trends**
    - ❌ Friday → **worst performance** (~28+ min)
    - ✅ Tuesday/Thursday → **best performance** (~26 min)

    > *"End-of-week demand increases delivery delays."*

    ---

    ## **🧹 10. Data Quality Insight**
    - Semi-Urban data showed **unrealistic patterns**  
    - Constant delivery time detected (~50 min)

    👉 **Cleaned to ensure reliable analysis**

    > *"Data quality is critical for predictive accuracy."*

    ---

    ## **❌ 11. Weak / Non-impact Features**
    - Order Type → **no significant effect** on delivery time
    - Preparation Time → **negligible impact**

    > *"Logistics dominate delivery time, not product characteristics."*

    **Implication:** Focus optimization on logistics, not restaurant factors

    ---

    ## **🧠 Final Conclusion**

    Delivery efficiency is mainly influenced by (in priority order):

    ### **Priority 1: Traffic (PRIMARY FACTOR)** 🚦
    - Most controllable through dynamic routing
    - Real-time traffic optimization is essential

    ### **Priority 2: Time of Day (CONTEXTUAL AMPLIFIER)** ⏰
    - Peak hours require special handling
    - Evening demands more sophisticated algorithms

    ### **Priority 3: Operational Load (NON-LINEAR IMPACT)** 📦
    - Route optimization becomes critical with multiple deliveries
    - Intelligent batching can reduce delays

    ### **Secondary Factors:**
    - **Distance:** Unavoidable but predictable
    - **Weather:** Moderate impact, requires adaptive strategies
    - **Partner Quality:** High-rated partners deliver better results
    - **Vehicle Type:** Impacts speed but not always efficiency

    ---

    ## **💡 Actionable Recommendations**

    1. **Implement real-time traffic-aware routing** 🚗
       - Reduces jam impact significantly
       - Can save 5-10 minutes per delivery in peak hours

    2. **Optimize time-of-day operations** ⏰
       - Increase delivery capacity during peak hours
       - Consider surge pricing/incentives for evening slots

    3. **Focus on partner quality** 🚴
       - Train riders on route efficiency, not just speed
       - Reward high-rated partners with better routes

    4. **Smart order batching** 📦
       - Limit multiple deliveries during peak times
       - Optimize routes to minimize non-linearity

    5. **Weather-adaptive strategies** 🌦️
       - Increase buffer time for poor weather
       - Dynamic pricing for fog/low-visibility conditions

    👉 **Optimizing these can significantly improve delivery performance.**
    """)

# ================================================================
# STRATEGIC INSIGHTS AT THE END
# ================================================================
st.divider()