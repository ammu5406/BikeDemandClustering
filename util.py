import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

class BikeModel:
    def __init__(self):
        self.df = None
        self.kmeans = None

    def load_and_train(self, path):
        # Load dataset
        df = pd.read_csv(path)

        # Convert seasons & weather
        df["season_name"] = df["season"].map({
            1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"
        })

        df["weather_name"] = df["weathersit"].map({
            1: "Clear",
            2: "Mist + Cloudy",
            3: "Light Snow / Rain",
            4: "Heavy Rain / Snow"
        })

        # Encode labels
        le_season = LabelEncoder()
        le_weather = LabelEncoder()

        df["season_encoded"] = le_season.fit_transform(df["season_name"])
        df["weather_encoded"] = le_weather.fit_transform(df["weather_name"])

        # Clustering features (removed atemp to match predictor inputs)
        features = df[[
            "temp", "hum", "windspeed",
            "season_encoded", "weather_encoded", "hr"
        ]]

        # Train KMeans
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        df["cluster"] = self.kmeans.fit_predict(features)

        # PCA for visualization components
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(features)
        df["pca_x"] = pca_res[:, 0]
        df["pca_y"] = pca_res[:, 1]

        self.df = df

    def get_clustered_df(self):
        return self.df

    def predict_cluster(self, temp, hum, windspeed, season, weather, hr):
        season_map = {"Spring": 0, "Summer": 3, "Fall": 1, "Winter": 2}
        weather_map = {"Clear": 0, "Mist + Cloudy": 1,
                       "Light Snow / Rain": 2, "Heavy Rain / Snow": 3}

        season_encoded = season_map[season]
        weather_encoded = weather_map[weather]

        cluster_row = [[temp, hum, windspeed, season_encoded, weather_encoded, hr]]
        cluster = self.kmeans.predict(cluster_row)[0]

        return int(cluster)
