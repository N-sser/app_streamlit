from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def preprocessing(df, features: list):
    X = df[features].copy()
    numerical = X.select_dtypes(include=['int64']).columns
    categorical = X.select_dtypes(include=['object']).columns

    # Scaling numerical variables
    ss = StandardScaler()
    X[numerical] = ss.fit_transform(X[numerical])

    # Encoding categorical variables
    le = LabelEncoder()
    X[categorical] = X[categorical].apply(le.fit_transform)
    return X

def clustering(X, n_clusters, linkage):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    y_hc = hc.fit_predict(X)
    return y_hc

def metrics(X, y_hc):
    return silhouette_score(X, y_hc), davies_bouldin_score(X, y_hc)