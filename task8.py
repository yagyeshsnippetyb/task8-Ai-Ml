import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Player': ['Virat', 'Rohit', 'Rahul', 'Dhawan', 'Bumrah', 'Shami', 'Jadeja', 'Hardik', 'Kuldeep', 'Chahal'],
    'Runs': [12000, 10500, 7000, 6000, 200, 300, 2500, 4000, 500, 400],
    'Wickets': [4, 8, 2, 1, 320, 280, 200, 60, 180, 190]
}

df = pd.DataFrame(data)

X = df[['Runs', 'Wickets']]

kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(df['Runs'], df['Wickets'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Runs')
plt.ylabel('Wickets')
plt.title('K-Means Clustering of Cricketers')
plt.show()
