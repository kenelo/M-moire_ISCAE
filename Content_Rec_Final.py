#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Importation des librairies

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud


from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importation du fichier SpotifyFeatures.csv sous forme de dataframe et affichage des 10 premières lignes


# In[3]:



data = pd.read_csv('SpotifyFeatures.csv', encoding='ISO-8859-1')
data.head(10)


# In[4]:


#infos sur les données


# In[5]:


data.info


# In[6]:


#Notre jeu de données contient 232725 lignes et 18 colonnes


# In[7]:


#Vérification des types de données


# In[8]:


data.dtypes


# In[9]:


#statistiques descriptives


# In[10]:


data.describe()


# In[11]:


#Les corrélations en heatmap


# In[12]:


sns.heatmap(data.corr(),cmap="YlOrRd")


# In[13]:


#forte corrélation positive entre energy et loudness
#forte corrélation négative entre: 1- accousticness et loudness 2-energy et accousticness


# In[14]:


#Vérification des veleurs manquantes


# In[15]:


data.isnull().any()


# In[16]:


#Notre jeu de données ne contient pas de valeurs manquantes


# In[17]:


#vérification  des doublons


# In[18]:


data = data.drop_duplicates()
data.shape


# In[19]:


#Notre jeu de données ne contient pas de doublons


# In[20]:


#Nombre de morceaux musicaux étudiés


# In[21]:


data['track_name'].nunique()


# In[22]:


#Nombre d'artistes étudiés


# In[23]:


data['artist_name'].nunique()


# In[24]:


#Artistes les plus écoutés


# In[25]:


artistes = data['artist_name'].value_counts().reset_index().head(10)
print(artistes)


# In[26]:


#Visualisation des artistes les plus écoutés


# In[27]:


data['artist_name'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('Artiste')
plt.ylabel('Fréquence')
plt.title('Artistes les plus écoutés')


# In[28]:


#Liste des 10 morceaux musicaux les plus écoutés


# In[29]:


tracks = data['track_name'].value_counts().reset_index().head(10)
print(tracks)


# In[30]:


#Visualisation des morceaux les plus écoutés


# In[31]:


data['track_name'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('morceaux_musicaux')
plt.ylabel('Fréquence')
plt.title('morceaux les plus écoutés')


# In[32]:


#Nombre de genres musicaux


# In[33]:


data['genre'].nunique()


# In[34]:


#Visualisation des différents genres musicaux


# In[35]:


genres_col = data[['genre']].columns.values
for col in genres_col:
    df = data.groupby([col]).size().reset_index(name='count')
    plt.figure(figsize=(18,8))
    plt.xticks(rotation=45)
    sns.set_style("ticks")
    sns.barplot(data = df, x= col, y= 'count')


# In[36]:


#Visualisation des genres les plus dominants


# In[37]:


wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150,
                      background_color='white').generate(" ".join(data.genre))

plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[38]:


#Visualisation du mode


# In[39]:


mode_col = data[['mode']].columns.values
for col in mode_col:
    dff = data.groupby([col]).size().reset_index(name='count')
    plt.figure(figsize=(18,8))
    plt.xticks(rotation=45)
    sns.set_style("ticks")
    sns.barplot(data = dff, x= col, y= 'count')


# In[40]:


#La plupart des morceaux musicaux sont en mode Major


# In[41]:


#Corrélations des 4 variables


# In[42]:


Correlation=data[['acousticness','energy','valence','loudness']]


# In[43]:


sns.heatmap(Correlation.corr(),annot=True,cmap="YlOrRd")


# In[44]:


#Visualisation de la relation entre  energie et bruit


# In[45]:


sns.jointplot(x=data['loudness'], y=data['energy'], data=data, kind="kde", color='lightblue');


# In[46]:


#La distribution de l'attribut popularité


# In[47]:


plt.hist(data['popularity'],bins=100)

plt.show()


# In[48]:


#Il existe des morceaux musicaux dont la popularité=0


# In[49]:


#Préparation des données


# In[50]:


#Attributs sélectionnés : mode, acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechness, tempo, valence
#Attributs supprimés : popularity, genre, key, mode, time_signature

data_features = data.drop(data.columns[[0, 1, 2, 3, 4, 10, 13, 16]], axis =1)

data_features.head()


# In[51]:


#Normalization des données pour avoir des valeurs entre  0 et 1 pour effectuer une analyse en composante principale (ACP)


# In[52]:


from sklearn.decomposition import PCA


# In[53]:


# Normalisation des données avec Min/Max

data_norm = data_features

scaler = MinMaxScaler() 

data_norm = scaler.fit_transform(data_norm)

data_norm = pd.DataFrame(data_norm, columns = data_features.columns)

data_norm.describe()


# In[54]:


# Construction d'un histogramme pour montrer la difference 

plt.hist(data_features['tempo'], bins=10)     #données d'origine
plt.show()


plt.hist(data_norm['tempo'], bins=10)          #données standardisées
plt.show()


# In[55]:


#ACP pour réduire les colonnes des attributs


# In[56]:


#Ajuster l'algorithme ACP avec nos données

pca = PCA().fit(data_norm)


# In[57]:


#Somme cumulée de la variance expliquée 

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Nombre de Composants')
plt.ylabel('Variance (%)') #for each component
plt.title(' Variance expliquée données Spotify')
plt.show()


# In[58]:


#Affichage de la variance expliquée pour chaque composante

variance_expliquée = pca.explained_variance_ratio_

print(variance_expliquée)


# In[59]:


#Variance expliquée pour les 8 composantes

print('La variance expliquée pour ces composantes est:  ', variance_expliquée[0:8].sum())


# In[60]:


#Choix du nombre de composantes

pca = PCA(n_components = 8)

data_pca = pca.fit_transform(data_norm)


# In[61]:


#Analyse des attributs et leurs interdépendances 


# In[62]:


from pandas.plotting import scatter_matrix


# In[63]:


#Visualisation des relations entre tous les attributs 

scatter_matrix(data_norm)

plt.gcf().set_size_inches(30, 30)

plt.show()


# In[64]:


#Utilisation de la corrélation spearman pour measurer les relations entre attributs 

pd.set_option('display.width', 100)
pd.set_option('precision', 3)

correlation = data_norm.corr(method='spearman')

print(correlation)


# In[65]:


# heatmap des corrélations pour visualiser les relations entre attributs 

plt.figure(figsize=(10,10))
plt.title('Correlation heatmap')

sns.heatmap(correlation, annot = True, vmin=-1, vmax=1, cmap="YlGnBu", center=1)


# In[66]:


#Partie3 - Recommandations de musique
#Former des Playlists en se basant sur les charactéristiques d'attributs en utilisant les techniques de Machine Learning  K-Means Clustering


# In[67]:


#Normalisation des données pour fixer le skew et avoir  mean=0 et std=1 afin de faire du clustering 


# In[68]:


#Trouver le skew pour chaque attribut

skew = data_features.skew()

print(skew)


# In[69]:


#Normalization des données

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data_features)

data_scaled = pd.DataFrame(data_scaled)

data_scaled.head()


# In[70]:


#La différence en histogramme

plt.hist(data_features['tempo'], bins=10)                    #données d'origine
plt.show()

plt.hist(data_scaled.iloc[8], bins=10)                            #données normalisées
plt.show()


# In[71]:


#Trouver le nombre de clusters approprié


# In[72]:


#Trouver le meilleur nombre de clusters en utilisant la méthode elbow et inértie


k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
inertias = []

for i in k:
    km = KMeans(n_clusters=i, max_iter=1000, random_state=42)
    km.fit(data_scaled)
    inertias.append(km.inertia_)

plt.plot(k, inertias)
plt.xlabel("Valeurs pour k")
plt.ylabel("Inerties")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

plt.show()


# In[73]:


k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

score=[]

for n_cluster in k:
    kmeans = KMeans(n_clusters=n_cluster).fit(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
    score.append(silhouette_score(data_scaled, kmeans.labels_))
    
    print('Score Silouhette pour %i Clusters: %0.4f' % (n_cluster, silhouette_avg))


# In[74]:


#Visualisation des options de clusters

plt.plot(k, score, 'o-')
plt.xlabel("Valeurs pour k")
plt.ylabel("Score Silhouette ")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

plt.show()


# In[75]:


#Fixation du  nombre de clusters
kclusters = 8


#Exécution du k-means clustering
kmeans = KMeans(n_clusters=kclusters, init='k-means++', random_state=42).fit(data_scaled)

#Vérification des titres de clusters  
kmeans.labels_[0:13]


# In[76]:


#Ajout des titres de clusters au tableau
data.insert(0, 'Numéro Playlist', kmeans.labels_)


# In[77]:




data.head()


# In[79]:


#Playlist 1


# In[80]:


data.loc[data['Numéro Playlist'] == 0, data.columns[[3, 4]]]


# In[81]:


#Playlist 2


# In[82]:


data.loc[data['Numéro Playlist'] == 1, data.columns[[3, 4]]]


# In[83]:


#Playlist 3


# In[84]:


data.loc[data['Numéro Playlist'] == 2, data.columns[[3, 4]]]


# In[85]:


#Playlist 4


# In[86]:


data.loc[data['Numéro Playlist'] == 3, data.columns[[3, 4]]]


# In[87]:


#Playlist 5


# In[88]:


data.loc[data['Numéro Playlist'] == 6, data.columns[[3, 4]]]


# In[89]:


#Playlist 6


# In[90]:


data.loc[data['Numéro Playlist'] == 7, data.columns[[3, 4]]]


# In[91]:


#Playlist 7


# In[92]:


data.loc[data['Numéro Playlist'] == 8, data.columns[[3, 4]]]


# In[ ]:




