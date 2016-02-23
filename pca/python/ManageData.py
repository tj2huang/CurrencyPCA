import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

path = __file__[:__file__.find('python')]
COMPONENTS = 5


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
    transformed = pca.transform(data)
    df_transformed = pd.DataFrame(data=transformed,  index=data.index)
    return df_transformed


def plot(df_transformed):
    pdf = PdfPages(path + "graph.pdf")
    plt.figure()
    df_transformed.plot()
    pdf.savefig()
    plt.close()
    pdf.close()

parsed_data = setup(path + '/data/USD crosses.csv')
outliers = [filter(lambda x: abs(x[1]) >= 5, zip(parsed_data.index, parsed_data.ix[:, col])) for col in parsed_data.columns]
ls_outliers = [pd.Series(data=[a[1] for a in x], index=[a[0] for a in x]) for x in outliers]
plot(get_components(parsed_data))
