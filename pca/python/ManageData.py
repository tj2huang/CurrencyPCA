import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

path = __file__[:__file__.find('python')]
COMPONENTS = 5
OUTLIER_PERCENT = 7


def setup(data_path):
    """

    :param data_path:
    :return: dataframe
    """
    data = pd.read_csv(data_path, index_col='Date')
    data.index = data.index.astype(str)
    data.index = pd.to_datetime(data.index.str[:4].values + '-' + data.index.str[4:6].values +
                                '-' + data.index.str[6:].values)
    return data


def get_components(data):
    pca = PCA(n_components=COMPONENTS)
    pca.fit(data)
    print(pca.components_, pca.explained_variance_ratio_)
    # pca.components_ = ortho_rotation(pca.components_)
    transformed = pca.transform(data)
    df_transformed = pd.DataFrame(data=transformed,  index=data.index)
    return df_transformed


def plot(df_transformed):
    pdf = PdfPages(path + "graph.pdf")
    plt.figure()
    df_transformed.resample('W').plot()
    pdf.savefig()
    plt.close()
    pdf.close()


def ortho_rotation(lam, method='varimax',gamma=None,eps=1e-6, itermax=100):
    if gamma is None:
        if method == 'varimax':
            gamma = 1.0
        if method == 'quartimax':
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R

parsed_data = setup(path + '/data/USD crosses.csv')
outliers = [filter(lambda x: abs(x[1]) >= OUTLIER_PERCENT, zip(parsed_data.index, parsed_data.ix[:, col]))
            for col in parsed_data.columns]
ls_outliers = [pd.Series(data=[a[1] for a in x], index=[a[0] for a in x]) for x in outliers]
merged = pd.concat(ls_outliers)
parsed_data.drop(merged.index)
plot(get_components(parsed_data))
