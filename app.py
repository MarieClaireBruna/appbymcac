import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static

if st.button("🔄 Rafraîchir les données"):
    st.experimental_rerun()
    
#Charger le jeu de donnee
df = pd.read_csv('AmesHousing.csv', sep=',', encoding='utf-8')
df.head()

#Nettoyage des donnees
print(df.isnull().sum()) #Verifie les valeurs manquantes
df = df.dropna(axis=1, thresh=int(0.7 * len(df))) #supprime les colonnes avec trop de donnee manquantes
df = df[df["SalePrice"] < df['SalePrice'].quantile(0.99)]

print(df['SalePrice'].describe()) #description statistique moyenne std etc


#INTERFACE STREAMLIT
st.title('Dashboard Immobilier by Claire')
st.write('Nous allons analyser les donnees qui nous ont ete soumises')
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Tableau de Bord</h1>", unsafe_allow_html=True)
st.write(df.head()) #Pour un apercu des donnees


st.metric(label="🏡 Nombre total de ventes", value=df.shape[0])


#Filtre sur les prix 
prix_min, prix_max = int(df['SalePrice'].min()), int(df['SalePrice'].max())
prix_filter = st.slider('Veuillez selectionner une plage de prix', prix_min, prix_max, (prix_min, prix_max))
filtered_df = df[(df["SalePrice"] >= prix_filter[0]) & (df["SalePrice"] <= prix_filter[1])]

#Nous allons utiliser un histogramme pour le diagramme des prix
st.subheader('Distribution des prix')
fig, ax = plt.subplots()
sns.histplot(filtered_df['SalePrice'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

#Selection de la localisation
locations = df["Neighborhood"].unique()  # Liste des quartiers uniques
selected_location = st.selectbox("Sélectionnez une localisation", ["Tous"] + list(locations))

# Filtrer par année de construction
min_year, max_year = int(df["Year Built"].min()), int(df["Year Built"].max())
selected_year = st.slider("Sélectionnez une année de construction", min_year, max_year, (min_year, max_year))


# Appliquer les filtres
with st.sidebar:
    st.header("Filtres")
    year_filter = st.slider("Année de construction", int(df["Year Built"].min()), int(df["Year Built"].max()), (2000, 2020))
    location_filter = st.multiselect("Localisation", df["Neighborhood"].unique())
    df_filtered = df[(df["Year Built"].between(*year_filter)) & (df["Neighborhood"].isin(location_filter))]
#filtered_df = df.copy()

if selected_location != "Tous":
    filtered_df = filtered_df[filtered_df["Neighborhood"] == selected_location]

filtered_df = filtered_df[(filtered_df["Year Built"] >= selected_year[0]) & (filtered_df["Year Built"] <= selected_year[1])]


# Afficher les données filtrées
st.write(f"Nombre de biens correspondants : {len(filtered_df)}")
st.write(filtered_df.head())

# Afficher la distribution des prix après filtrage
st.subheader("Distribution des prix après filtrage")
fig, ax = plt.subplots()
sns.histplot(filtered_df["SalePrice"], bins=30, kde=True, ax=ax)
st.pyplot(fig)


st.subheader("Carte de densité des ventes par région")
# Vérifier si les colonnes Latitude et Longitude existent
if "Latitude" in df.columns and "Longitude" in df.columns:
    # Création de la carte centrée sur les coordonnées moyennes
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    sales_map = folium.Map(location=map_center, zoom_start=10)

    # Ajouter les points sur la carte
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            popup=f"Prix: {row['SalePrice']}",
            color="blue",
            fill=True,
            fill_color="blue"
        ).add_to(sales_map)

    # Affichage dans Streamlit
    folium_static(sales_map)
else:
    st.warning("Les colonnes Latitude et Longitude sont absentes du dataset.")
    
    
    st.subheader("Ventes par année de construction")
sales_per_year = df.groupby("Year Built")["SalePrice"].count().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=sales_per_year, x="Year Built", y="SalePrice", ax=ax)
plt.xticks(rotation=90)
ax.set_xlabel("Année de construction")
ax.set_ylabel("Nombre de ventes")
st.pyplot(fig)

st.subheader("Corrélations entre caractéristiques")
correlation_matrix = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

#Nous allons predire le prix d'un bien immobilier 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#on selectionne les variables
x = df[['Gr Liv Area', 'Year Built']]
y = df['SalePrice']
#Division en train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=40)
#Model de regression
model = LinearRegression()
model.fit(x_train, y_train)
#Prediction
predictions = model.predict(x_test)
#Et enfin on affiche les resultats 
st.write("Voici mes predictions de prix basees sur la surface et l'annee de construction :")
st.write(predictions[:10])