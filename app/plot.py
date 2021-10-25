from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from app.utils import loggable

@loggable
class Plotter:
    def __init__(self, options) -> None:
        plot_path = '../plots' if 'plot_path' not in options else options['plot_path']
        self.session_id = datetime.now().strftime('%Y-%m-%d_%H_%M_%s')
        self.plot_path = os.path.join(plot_path, self.session_id)
        self.df = None

    def plot_components(self, enc_output, labels):
        os.mkdir(self.plot_path)
        self.log('data preparing')

        self.df = pd.DataFrame({'labels': labels})
        self.log('Reduce dimensionality')
        self.determine_components(enc_output)
        self.log('2 Comp plot')
        self.two_component_plot()
        self.log('3 Comp plot')
        self.three_component_plot()



    def two_component_plot(self):
        plt.figure(figsize=(16,7))
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('PCA, 2 components')
        sns.scatterplot(
            x="pca-1", y="pca-2", hue="labels",
            palette=sns.color_palette("hls", 10),
            data=self.df,
            legend="full",
            alpha=0.3,
            ax=ax1
        )

        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('tSNE, 2 components')
        sns.scatterplot(
            x="tsne-1", y="tsne-2", hue="labels",
            palette=sns.color_palette("hls", 10),
            data=self.df,
            legend="full",
            alpha=0.3,
            ax=ax2
        )
        plt.savefig(os.path.join(self.plot_path, 'two_component_plot.png'))

    def three_component_plot(self):
        df = self.df
        rndperm = np.random.permutation(df.shape[0])
        colors = {0:'tab:blue', 1:'tab:orange'}
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.set_title('PCA, 3 components')
        ax.scatter(
            xs=df.loc[rndperm,:]["pca-1"], 
            ys=df.loc[rndperm,:]["pca-2"], 
            zs=df.loc[rndperm,:]["pca-3"], 
            c=df.loc[rndperm,:]["labels"].map(colors), 
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three') 
        
        plt.savefig(os.path.join(self.plot_path, 'two_component_plot.png'))

    def determine_components(self, enc_output):
        df = self.df
        self.log('PCA calculation')
        pca = PCA(n_components=3)
        result_pca = pca.fit_transform(enc_output)
        df['pca-1'] = result_pca[:, 0]
        df['pca-2'] = result_pca[:, 2]
        df['pca-3'] = result_pca[:, 3]
        self.log('Explained variation per PC', pca.explained_variance_ratio_)
        self.log('t-SNE calculation')
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        results_tsne = tsne.fit_transform(enc_output)
        df['tsne-1'] = results_tsne[:,0]
        df['tsne-2'] = results_tsne[:,1]
        self.log('calculations finished')
