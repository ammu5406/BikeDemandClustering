import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from util import BikeModel

st.set_page_config(page_title="Bike Demand Pattern Clustering Dashboard", layout="wide")

# ----------- 1. LOAD MODEL & DATA -----------
@st.cache_resource
def load_model():
    model = BikeModel()
    model.load_and_train("data/hour.csv")
    return model

bike_model = load_model()
df = bike_model.get_clustered_df()

st.title("Bike Demand Pattern Clustering Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Interactive Predictor",
    "Clustering & Patterns",
    "Data Preview & EDA"
])

# ----------- Tab 1: Interactive Predictor -----------
with tab1:
    st.header("Predict Bike Demand Pattern")
    st.markdown("Use the model to find the closest demand cluster based on environmental conditions.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        weather = st.selectbox("Weather", ["Clear", "Mist + Cloudy", "Light Snow / Rain", "Heavy Rain / Snow"])
    with c2:
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
        hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
    with c3:
        windspeed = st.slider("Windspeed (normalized)", 0.0, 1.0, 0.3)
        hr = st.slider("Hour of day", 0, 23, 12)

    if st.button("Predict Cluster"):
        cluster = bike_model.predict_cluster(temp, hum, windspeed, season, weather, hr)
        st.info(f"Belongs to K-Means Cluster: **{cluster}**")
        
    st.markdown("---")
    st.subheader("Cluster Explanations")
    st.markdown("""
    - **Cluster 0 (Evening Rush / High Demand)**: Typically peaks around 8:00 PM with moderate-to-warm temperatures. Represents heavy evening activity with an **average demand of ~260 bikes/hour**.
    - **Cluster 1 (Late Night / Low Demand)**: Characterized by early AM hours (e.g., 2:00 AM) with cooler temperatures and high humidity. Represents extremely low activity with an **average demand of ~26 bikes/hour**.
    - **Cluster 2 (Afternoon / Favorable Weather)**: Occurs around 1:00 PM - 2:00 PM during warm, dry, and pleasant weather conditions resulting in high continuous rentals. Shows an **average demand of ~253 bikes/hour**.
    - **Cluster 3 (Morning Commute / Moderate Demand)**: Peaks around 7:00 AM - 8:00 AM representing the morning commute cycle in cooler and more humid morning baseline conditions. Sees an **average demand of ~177 bikes/hour**.
    """)

# ----------- Tab 2: Clustering -----------
with tab2:
    st.header("8. Clustering (KMeans + PCA plot)")
    st.markdown("We apply K-Means clustering (K=4) to group similar demand behaviors based on weather and hour variables.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_pca = px.scatter(
            df, x="pca_x", y="pca_y", color=df["cluster"].astype(str),
            title="2D PCA Projection of Bike Demand Clusters",
            hover_data=["hr", "season_name", "weather_name", "cnt"],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col2:
        st.subheader("Cluster Profiles")
        cluster_summary = df.groupby("cluster")[["cnt", "temp", "hum"]].mean().reset_index()
        st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"))
        st.info("Clusters typically separate high demand rush hours from low demand night hours, as well as favorable versus poor weather.")

# ----------- Tab 3: EDA -----------
with tab3:
    st.header("2. Load & Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.header("4. Data Cleaning")
    st.markdown("""
    - Extracted & mapped **Season** and **Weather** integers to categories.
    - Encoded categorical variables using `LabelEncoder`.
    - Features selected: `temp`, `hum`, `windspeed`, `season_name`, `weather_name`, `hr`.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("5. Univariate: Demand Distribution")
        fig1 = px.histogram(df, x="cnt", nbins=50, title="Distribution of Total Rentals (cnt)",
                            color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("This histogram highlights the overall distribution of rented bikes. Most hours see lower demand volumes, with extreme high-demand instances being rarer.")
        
    with col2:
        st.subheader("5. Univariate: Weather Conditions")
        fig2 = px.pie(df, names='weather_name', title="Proportion of Weather Conditions", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("The majority of recorded observations feature clear weather, which heavily influences baseline rider volumes.")
        
    st.header("6. Bivariate Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Demand vs Hour of Day")
        fig3 = px.box(df, x="hr", y="cnt", color="season_name", title="Hourly Rentals by Season")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Boxplots visualize the spread of demand across the hour of the day. You can easily spot the dual peaks corresponding to morning and evening rush hours.")
        
    with col4:
        st.subheader("Demand vs Temperature")
        fig4 = px.scatter(df, x="temp", y="cnt", color="season_name", opacity=0.5,
                          title="Temperature and Rentals")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("There is a general upward trend showing that warmer temperatures correlate with higher bike rentals, peaking in the summer and dipping in winter.")

    st.header("7. Demand by Features")
    st.markdown("Average demand, temperature, and humidity based on the combination of season and weather conditions:")
    agg_df = df.groupby(["season_name", "weather_name"])[["cnt", "temp", "hum"]].mean().reset_index()
    st.dataframe(agg_df.style.background_gradient(cmap="Purples"), use_container_width=True)
