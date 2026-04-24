import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, TransformerMixin

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Delivery Time Analytics & Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM STYLING ========================
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
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin: 20px 0;
}
.perf-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 8px 0;
}
.perf-label {
    font-size: 12px;
    opacity: 0.9;
    margin-bottom: 5px;
}
.perf-value {
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================== HELPER FUNCTIONS ========================

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
    
    # Rating groups (from Delivery_person_Ratings)
    if 'Delivery_person_Ratings' in df.columns:
        df['Delivery_Person_Rating_Group'] = pd.cut(
            df['Delivery_person_Ratings'],
            bins=[0, 3, 4, 5],
            labels=['Low (0-3)', 'Medium (3-4)', 'High (4-5)'],
            include_lowest=True
        )
    
    # Age groups (from Delivery_person_Age)
    if 'Delivery_person_Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Delivery_person_Age'],
            bins=[14, 25, 35, 50],
            labels=['Young (≤25)', 'Adult (25-35)', 'Senior (>35)'],
            include_lowest=True
        )
    
    # Traffic-distance interaction
    df['distance_traffic'] = df['distance_km'] * df['Road_traffic_density'].map(
        {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    )
    
    return df


# ======================== FEATURE ENGINEERING CLASS ========================

class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.rating_bins = None

    def fit(self, X, y=None):
        X = X.copy()
        self.rating_bins = pd.qcut(
            X['Delivery_person_Ratings'],
            q=3,
            retbins=True,
            duplicates='drop'
        )[1]
        return self

    def transform(self, X):
        X = X.copy()
        X['Order_Datetime'] = pd.to_datetime(X['Order_Datetime'], errors='coerce')
        X['Pickup_Datetime'] = pd.to_datetime(X['Pickup_Datetime'], errors='coerce')

        X["distance_km"] = haversine(
            X["Restaurant_latitude"],
            X["Restaurant_longitude"],
            X["Delivery_location_latitude"],
            X["Delivery_location_longitude"]
        )

        X["delivery_rating_group"] = pd.cut(
            X['Delivery_person_Ratings'],
            bins=self.rating_bins,
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )

        X["age_group"] = pd.cut(
            X['Delivery_person_Age'],
            bins=[14, 25, 35, 60],
            labels=['Young', 'Adult', 'Senior']
        )

        X["distance_group"] = pd.cut(
            X['distance_km'],
            bins=[0, 5, 10, 25],
            labels=['Short Distance', 'Medium Distance', 'Long Distance']
        )

        X['Prep_Time(min)'] = (
            X['Pickup_Datetime'] - X['Order_Datetime']
        ).dt.total_seconds() / 60

        X['Order_hour'] = X['Order_Datetime'].dt.hour
        X['Order_day'] = X['Order_Datetime'].dt.day_name()
        X['isWeekend'] = X['Order_day'].isin(["Saturday", "Sunday"]).astype(int)

        X['Time_Of_Day'] = pd.cut(
            X['Order_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True
        )

        X = X.drop(columns=['Order_Datetime', 'Pickup_Datetime'], errors='ignore')
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


# ======================== DATA LOADING ========================

@st.cache_data
def load_and_preprocess():
    """Load and preprocess delivery dataset."""
    df = pd.read_csv("data/Cleaned Delivery Dataset.csv")
    df = engineer_features(df)
    return df


@st.cache_resource
def load_model():
    """Load the trained prediction model."""
    try:
        with open("models/pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# ======================== MAIN APP NAVIGATION ========================

def main():
    # Navigation tabs at the top
    tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🔮 Prediction Engine"])
    
    # ================================================================
    # TAB 1 — ANALYTICS DASHBOARD
    # ================================================================
    with tab1:
        analytics_dashboard()
    
    # ================================================================
    # TAB 2 — PREDICTION ENGINE
    # ================================================================
    with tab2:
        prediction_engine()


def analytics_dashboard():
    """Analytics Dashboard - Data Exploration and Visualization"""
    
    # Load data
    df = load_and_preprocess()
    
    # Page header
    st.title("📊 Delivery Time Analytics Dashboard")
    st.caption("Interactive dashboard analyzing delivery performance factors")
    
    # ======================== SIDEBAR FILTERS ========================
    st.sidebar.header("🔎 Filters")

    cities = sorted(df["City"].unique())
    zones = sorted(df["Zone"].unique())
    traffic_levels = sorted(df["Road_traffic_density"].unique())
    weather_types = sorted(df["Weather_conditions"].unique())
    time_periods = ["Night", "Morning", "Afternoon", "Evening"]
    vehicle_types = sorted(df["Type_of_vehicle"].unique())
    city_types = sorted(df["City_Type"].unique())

    sel_cities = st.sidebar.multiselect("City", cities, default=cities)
    sel_zones = st.sidebar.multiselect("Zone", zones, default=zones)
    sel_traffic = st.sidebar.multiselect("Traffic Density", traffic_levels, default=traffic_levels)
    sel_weather = st.sidebar.multiselect("Weather", weather_types, default=weather_types)
    sel_time = st.sidebar.multiselect("Time of Day", time_periods, default=time_periods)
    sel_vehicle = st.sidebar.multiselect("Vehicle Type", vehicle_types, default=vehicle_types)
    sel_city_type = st.sidebar.multiselect("City Type", city_types, default=city_types)

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
            rating_time = fdf.groupby("Delivery_Person_Rating_Group", observed=True)["Time_taken (min)"].mean().reset_index()
            fig = px.bar(rating_time, x="Delivery_Person_Rating_Group", y="Time_taken (min)",
                        color="Time_taken (min)", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            rating_speed = fdf.groupby("Delivery_Person_Rating_Group", observed=True)["speed_kmph"].mean().reset_index()
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
            age_delivery = fdf.groupby("Age_Group", observed=True)["Time_taken (min)"].mean().reset_index()
            fig = px.bar(age_delivery, x="Age_Group", y="Time_taken (min)",
                        color="Time_taken (min)", color_continuous_scale="Oranges")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            age_speed = fdf.groupby("Age_Group", observed=True)["speed_kmph"].mean().reset_index()
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
        rating_age = fdf.groupby(["Age_Group", "Delivery_Person_Rating_Group"], observed=True)["Time_taken (min)"].mean().reset_index()
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




def prediction_engine():
    """Prediction Engine - ML-based Delivery Time Forecasting"""
    
    # Load model and data
    pipeline = load_model()
    
    # Load and cache data for getting unique values
    @st.cache_data
    def load_prediction_data():
        try:
            df = pd.read_csv("data/Cleaned Delivery Dataset.csv")
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found. Please ensure 'Cleaned Delivery Dataset.csv' exists in data folder.")
            return None
    
    data = load_prediction_data()
    
    # Page header
    st.title("🔮 Delivery Time Prediction Engine")
    st.markdown("Predict delivery times using machine learning")
    
    # Sidebar - Information and Instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.info("""
        1. Fill in the delivery details in the main panel
        2. The app will predict the estimated delivery time
        3. Check the prediction metrics for accuracy info
        """)
        
        st.divider()
        st.header("📊 Model Info")
        st.write("""
        - **Model Type:** XGBRegressor
        - **Features:** 30+
        - **Target:** Delivery Time (minutes)
        """)
        
        st.divider()
        st.header("📈 Model Performance")
        
        # Performance metrics from training
        metrics_data = {
            'R² Score': 0.8312,
            'Adjusted R²': 0.8293,
            'MAE (min)': 3.0226,
            'RMSE (min)': 3.7920,
            'MSE': 14.3790
        }
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-label">R² Score</div>
                <div class="perf-value">{metrics_data['R² Score']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-label">MAE (minutes)</div>
                <div class="perf-value">{metrics_data['MAE (min)']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-label">MSE</div>
                <div class="perf-value">{metrics_data['MSE']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-label">Adjusted R²</div>
                <div class="perf-value">{metrics_data['Adjusted R²']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-label">RMSE (minutes)</div>
                <div class="perf-value">{metrics_data['RMSE (min)']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📋 Metrics Explanation"):
            st.write("""
            - **R² Score**: Explains 83.1% of variance in delivery time (closer to 1 is better)
            - **Adjusted R²**: R² adjusted for feature count
            - **MAE**: Average prediction error is ±3.02 minutes
            - **RMSE**: Root mean squared error accounts for larger errors
            - **MSE**: Mean squared error metric
            """)
        
        st.divider()
        st.header("⚙️ Feature Categories")
        with st.expander("Delivery Person"):
            st.write("Age, Ratings, Multiple Deliveries")
        with st.expander("Location"):
            st.write("Restaurant & Delivery Coordinates")
        with st.expander("Order Details"):
            st.write("Type, City, Zone, Weather")
        with st.expander("Vehicle & Traffic"):
            st.write("Vehicle Type, Condition, Traffic Density")
    
    # ======================== PREDICTION INTERFACE ========================
    if pipeline is not None and data is not None:
        # Get unique values from data for selectboxes
        cities = sorted(data["City"].unique().tolist())
        zones = sorted(data["Zone"].unique().tolist())
        weather_types = sorted(data["Weather_conditions"].unique().tolist())
        order_types = sorted(data["Type_of_order"].unique().tolist())
        vehicle_types = sorted(data["Type_of_vehicle"].unique().tolist())
        city_types = sorted(data["City_Type"].unique().tolist())
        vehicle_conditions = sorted(data["Vehicle_condition"].unique().tolist())
        traffic_levels = sorted(data["Road_traffic_density"].unique().tolist())
        
        col1, col2 = st.columns(2)
        
        # Left column - Delivery Person Details
        with col1:
            st.subheader("👤 Delivery Person")
            delivery_age = st.slider("Age", 18, 60, 30, key="pred_age")
            delivery_rating = st.slider("Rating (1-5)", 1.0, 5.0, 4.5, key="pred_rating")
            multiple_deliveries = st.number_input("Multiple Deliveries Count", 0, 10, 1, key="pred_multi_del")
        
        # Right column - Location Details
        with col2:
            st.subheader("📍 Location")
            col2a, col2b = st.columns(2)
            with col2a:
                st.write("Restaurant Coordinates")
                rest_lat = st.number_input("Restaurant Latitude", -90.0, 90.0, 12.971234, key="pred_rest_lat", format="%.6f")
                rest_lon = st.number_input("Restaurant Longitude", -180.0, 180.0, 77.712312, key="pred_rest_lon", format="%.6f")
            with col2b:
                st.write("Delivery Location Coordinates")
                delivery_lat = st.number_input("Delivery Latitude", -90.0, 90.0, 12.962345, key="pred_del_lat", format="%.6f")
                delivery_lon = st.number_input("Delivery Longitude", -180.0, 180.0, 77.682456, key="pred_del_lon", format="%.6f")
        
        # Order and Vehicle Details
        st.subheader("🛵 Order & Vehicle Details")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            city = st.selectbox("City", cities, key="pred_city")
            zone = st.selectbox("Zone", zones, key="pred_zone")
            city_type = st.selectbox("City Type", city_types, key="pred_city_type")
        
        with col4:
            weather = st.selectbox("Weather", weather_types, key="pred_weather")
            order_type = st.selectbox("Order Type", order_types, key="pred_order_type")
            vehicle_type = st.selectbox("Vehicle Type", vehicle_types, key="pred_vehicle")
        
        with col5:
            vehicle_condition = st.selectbox("Vehicle Condition", vehicle_conditions, key="pred_vehicle_cond")
            traffic_density = st.selectbox("Traffic Density", traffic_levels, key="pred_traffic")
            festival = st.selectbox("Festival", ["No", "Yes"], key="pred_festival")
        
        # Time Details
        st.subheader("⏰ Order Time")
        col6, col7 = st.columns(2)
        
        with col6:
            order_date = st.date_input("Order Date", key="pred_date")
        with col7:
            order_hour = st.slider("Order Hour", 0, 23, 12, key="pred_hour")
        
        order_datetime = pd.Timestamp(
            year=order_date.year,
            month=order_date.month,
            day=order_date.day,
            hour=order_hour
        )
        order_day = order_datetime.day_name()
        
        # Predict button
        if st.button("🔮 Predict Delivery Time", use_container_width=True, type="primary"):
            try:
                # Input validation
                if delivery_lat == rest_lat and delivery_lon == rest_lon:
                    st.warning("⚠️ Delivery location is same as restaurant. Please update coordinates.")
                    st.stop()
                
                # Create input DataFrame with all required columns
                input_data = pd.DataFrame({
                    'Delivery_person_Age': [delivery_age],
                    'Delivery_person_Ratings': [delivery_rating],
                    'multiple_deliveries': [multiple_deliveries],
                    'Restaurant_latitude': [rest_lat],
                    'Restaurant_longitude': [rest_lon],
                    'Delivery_location_latitude': [delivery_lat],
                    'Delivery_location_longitude': [delivery_lon],
                    'City': [city],
                    'Zone': [zone],
                    'Weather_conditions': [weather],
                    'Type_of_order': [order_type],
                    'Type_of_vehicle': [vehicle_type],
                    'City_Type': [city_type],
                    'Road_traffic_density': [traffic_density],
                    'Vehicle_condition': [vehicle_condition],
                    'Festival': [festival],
                    'Order_Datetime': [order_datetime],
                    'Pickup_Datetime': [order_datetime + pd.Timedelta(minutes=15)]
                })
                
                # Make prediction
                prediction = pipeline.predict(input_data)[0]
                
                # Calculate distance
                distance = haversine(rest_lat, rest_lon, delivery_lat, delivery_lon)
                
                # Calculate delivery time
                delivery_time = order_datetime + pd.Timedelta(minutes=float(prediction))
                
                # Display results
                st.divider()
                st.markdown(f"""
                <div class="prediction-box">
                ⏱️ Estimated Delivery Time<br>
                {prediction:.1f} minutes
                </div>
                """, unsafe_allow_html=True)
                
                # Key metrics in columns
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                
                with col_res1:
                    hours = int(prediction // 60)
                    minutes = int(prediction % 60)
                    st.metric("Duration", f"{hours}h {minutes}m")
                
                with col_res2:
                    st.metric("Expected Delivery", delivery_time.strftime("%H:%M"))
                
                with col_res3:
                    st.metric("Distance", f"{distance:.1f} km")
                
                with col_res4:
                    # Estimate cost/priority based on distance and time
                    if distance < 5 and prediction < 20:
                        priority = "🟢 Fast"
                    elif distance < 10 and prediction < 30:
                        priority = "🟡 Medium"
                    else:
                        priority = "🔴 Slow"
                    st.metric("Priority", priority)
                
                st.divider()
                
                # Detailed breakdown
                st.subheader("📊 Prediction Details & Analysis")
                
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.info(f"""
                    **📍 Route Information**
                    - Distance: {distance:.2f} km
                    - Estimated Speed: {(distance*60/prediction):.1f} km/h
                    - Time of Day: {order_datetime.strftime('%I:%M %p')}
                    - Day: {order_day}
                    """)
                
                with detail_col2:
                    st.warning(f"""
                    **🚚 Delivery Conditions**
                    - Traffic: {traffic_density}
                    - Weather: {weather}
                    - Vehicle: {vehicle_type}
                    - Condition: {vehicle_condition}
                    """)
                
                with detail_col3:
                    st.success(f"""
                    **👤 Delivery Partner Profile**
                    - Age: {delivery_age} years
                    - Rating: ⭐ {delivery_rating}/5.0
                    - Experience: {"Experienced" if delivery_rating >= 4.0 else "Good" if delivery_rating >= 3.0 else "Developing"}
                    - Multiple Orders: {multiple_deliveries}
                    """)
                
                st.divider()
                
                # Model confidence and factors
                st.subheader("🎯 Key Factors Affecting Delivery Time")
                
                factor_col1, factor_col2 = st.columns(2)
                
                with factor_col1:
                    st.markdown("""
                    **🔴 Primary Factors (Highest Impact):**
                    - **Traffic Density**: Most critical factor
                    - **Distance**: Strongly correlates with time
                    - **Multiple Deliveries**: Adds significant delays
                    """)
                
                with factor_col2:
                    st.markdown("""
                    **🟡 Secondary Factors:**
                    - **Time of Day**: Evening peak adds delays
                    - **Delivery Rating**: Higher ratings = faster delivery
                    - **Weather**: Impacts road conditions
                    """)
                
                # Prediction confidence based on model metrics
                st.divider()
                st.subheader("📈 Model Confidence & Accuracy")
                
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    confidence_percentage = 83.12  # Based on R² score
                    st.progress(confidence_percentage / 100, text=f"Model Accuracy: {confidence_percentage:.1f}%")
                    st.caption("Based on R² Score from training data")
                
                with conf_col2:
                    margin_of_error = 3.79  # RMSE
                    st.info(f"""
                    **Expected Margin of Error: ±{margin_of_error:.1f} minutes**
                    
                    *Actual time may vary by ±{margin_of_error:.1f} min due to:*
                    - Real-time traffic variations
                    - Unexpected route changes
                    - Order complexity
                    """)
                
                st.divider()
                
                # Order summary table
                st.subheader("📋 Complete Order Summary")
                
                summary_data = {
                    'Category': ['Delivery Person', 'Delivery Person', 'Delivery Person', 
                                'Route', 'Route', 'Route', 
                                'Order', 'Order', 'Order', 
                                'Conditions', 'Conditions', 'Conditions'],
                    'Attribute': ['Age', 'Rating', 'Multiple Deliveries',
                                 'Distance', 'From', 'To',
                                 'Type', 'Time', 'Day',
                                 'Traffic', 'Weather', 'Vehicle'],
                    'Value': [f"{delivery_age} years", 
                             f"⭐ {delivery_rating}/5.0", 
                             f"{multiple_deliveries}",
                             f"{distance:.2f} km",
                             f"{rest_lat:.4f}, {rest_lon:.4f}",
                             f"{delivery_lat:.4f}, {delivery_lon:.4f}",
                             order_type,
                             order_datetime.strftime('%I:%M %p'),
                             order_day,
                             traffic_density,
                             weather,
                             f"{vehicle_type} ({vehicle_condition})"]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Download prediction details
                st.divider()
                prediction_text = f"""
DELIVERY TIME PREDICTION REPORT
{'='*50}

ESTIMATED DELIVERY TIME: {prediction:.1f} minutes

KEY INFORMATION:
- Expected Delivery: {delivery_time.strftime('%I:%M %p on %A')}
- Distance: {distance:.2f} km
- Estimated Speed: {(distance*60/prediction):.1f} km/h

DELIVERY PERSON:
- Age: {delivery_age} years
- Rating: {delivery_rating}/5.0
- Multiple Deliveries: {multiple_deliveries}

DELIVERY CONDITIONS:
- Traffic Density: {traffic_density}
- Weather: {weather}
- Vehicle Type: {vehicle_type}
- Vehicle Condition: {vehicle_condition}

ORDER DETAILS:
- Type: {order_type}
- Zone: {zone}
- City: {city}
- Order Time: {order_datetime.strftime('%I:%M %p')}
- Festival: {festival}

MODEL ACCURACY:
- R² Score: 83.12%
- Expected Error Margin: ±3.79 minutes
- Confidence Level: High

IMPORTANT NOTE:
The prediction is based on historical data and may vary due to:
- Real-time traffic conditions
- Route optimization
- Weather changes
- Order complexity
"""
                
                st.download_button(
                    label="📥 Download Prediction Report",
                    data=prediction_text,
                    file_name=f"delivery_prediction_{order_datetime.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("Error details have been logged. Please check your input values.")
                st.info("""
                **Troubleshooting Tips:**
                - Ensure all fields are filled correctly
                - Check latitude/longitude values are within valid ranges
                - Verify model file 'pipeline.pkl' exists in the directory
                """)
    else:
        if pipeline is None:
            st.error("❌ Unable to load the model. Please ensure 'pipeline.pkl' is in the app directory.")
        if data is None:
            st.error("❌ Unable to load the data. Please ensure 'Cleaned Delivery Dataset.csv' exists in the data folder.")


# ======================== RUN MAIN APP ========================
if __name__ == "__main__":
    main()
