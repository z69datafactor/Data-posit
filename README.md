# Data-posit












.................................................................

#Perform dimensionality reduction.
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

pca = PCA(0.98, svd_solver="full")

x_train = train_df.drop('y', axis = 1)
y_train = train_df['y']

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(y_train, x_train, test_size=.20, random_state=1)


#PCA_df
n_comp = 12
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train_df.drop(["y"], axis=1))
pca2_results_test = pca.transform(test_df)


pca.n_components_
pca.explained_variance_ratio_


THE BELOW IS GIVING ME ERROR WITH [ValueError: Expected 2D array, got 1D array instead:
array=[107.87  90.7  105.73 ... 108.57 142.46 106.19].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.]ValueError: Expected 2D array, got 1D array instead:
array=[107.87  90.7  105.73 ... 108.57 142.46 106.19].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
pca_x_train = pd.DataFrame(pca.transform(x_train))
pca_x_val = pd.DataFrame(pca.transform(x_val))
pca_test = pd.DataFrame(pca.transform(test_df))
