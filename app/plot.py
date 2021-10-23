from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

def plot_components(enc_output, labels, options):
    plot_path = '../plots' if 'plot_path' not in options else options['plot_path']
    session_id = datetime.now().strftime('%Y-%m-%d_%H_%M_%s')
    plot_path = os.path.join(plot_path, session_id)
    os.mkdirs(plot_path)

    df = pd.DataFrame({'encodings':enc_output, 'labels': labels})
    determine_components(df,enc_output)
    two_component_plot(plot_path, df)
    three_component_plot(plot_path, df)



def two_component_plot(plot_path, df):
    plt.figure(figsize=(16,7))
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('PCA, 2 components')
    sns.scatterplot(
        x="pca-1", y="pca-2", hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('tSNE, 2 components')
    sns.scatterplot(
        x="tsne-1", y="tsne-2", hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
    plt.show()
    plt.savefig(os.path.join(plot_path, 'two_component_plot.png'))

def three_component_plot(plot_path, df):
    rndperm = np.random.permutation(df.shape[0])
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.set_title('PCA, 3 components')
    ax.scatter(
        xs=df.loc[rndperm,:]["pca-1"], 
        ys=df.loc[rndperm,:]["pca-2"], 
        zs=df.loc[rndperm,:]["pca-3"], 
        c=df.loc[rndperm,:]["labels"], 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three') 
    plt.show()
    plt.savefig(os.path.join(plot_path, 'two_component_plot.png'))

def determine_components(df, enc_output):
    pca = PCA(n_components=3)
    result_pca = pca.fit_transform(enc_output)
    df['pca-1'] = result_pca[:, 0]
    df['pca-2'] = result_pca[:, 2]
    df['pca-3'] = result_pca[:, 3]
    print('Explained variation per PC', pca.explained_variance_ratio_)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    results_tsne = tsne.fit_transform(enc_output)
    df['tsne-1'] = results_tsne[:,0]
    df['tsne-2'] = results_tsne[:,1]
