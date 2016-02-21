import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

path = __file__[:__file__.find('python')]

data = pd.read_csv(path + '/data/All crosses.csv', index_col='Date')
data.index = data.index.astype(str)
data.index = pd.to_datetime(data.index.str[:4].values + '-' + data.index.str[4:6].values +
                            '-' + data.index.str[6:].values)
pca = PCA(n_components=3)
pca.fit(data)
transformed = pca.transform(data)
# cumulative_returns = transformed
cumulative_returns = (1+transformed/100).cumprod(axis=1)
df_transformed = pd.DataFrame(data=cumulative_returns,  index=data.index).resample('10D', how='prod').add(-1)*100.
pdf = PdfPages(path + "graph.pdf")
plt.figure()
df_transformed[0].plot()
pdf.savefig()
plt.close()
pdf.close()
print('hi')
