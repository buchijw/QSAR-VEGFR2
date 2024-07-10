import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class PCA_2D:
    def __init__(self, data: np.array):
        self.pca = PCA(n_components=2)
        self.init_fit(data)
        pass

    def init_fit(self, data: np.array):
        self.pca.fit(data)
        print('Explained variance: ', self.pca.explained_variance_ratio_)

    def plot(self, data: np.array, c:np.array = None, cmap:str = 'crest', s=10, hull:bool = True, figsize:tuple = (10, 8), title:str = '', title_fontsize = 14, title_x = 0.45, title_y = 1.0, verbose = True):
        print('--- PCA 2D PLOT ---') if verbose else None
        print('Data shape: ', data.shape) if verbose else None
        pca_proj = self.pca.transform(data)
        pca_df = pd.DataFrame(pca_proj,columns=['x','y'])
        if c is not None:
            print('Coloring enabled.') if verbose else None
            print('Color shape: ', c.shape) if verbose else None
            pca_df = pd.concat([pca_df,pd.DataFrame({'c':c})],axis=1)
        if hull:
            print('Convex Hull enabled.') if verbose else None
            convex = ConvexHull(pca_proj)
        
        print('Plotting...') if verbose else None
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2,  width_ratios=(6, 1), height_ratios=(1, 6),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1,0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        if c is not None:
            cmap = sns.color_palette(cmap, as_cmap=True)
            points = ax.scatter(pca_df.x, pca_df.y,c=pca_df.c, s=s, cmap=cmap)
        else:
            points = ax.scatter(pca_df.x, pca_df.y, s=s)
        sns.histplot(data=pca_df,x='x',kde=True,ax=ax_histx)
        sns.histplot(data=pca_df,y='y',kde=True,ax=ax_histy)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if hull:
            for simplex in convex.simplices:
                ax.plot(pca_proj[simplex, 0], pca_proj[simplex, 1], 'k-',linewidth=1)
        if c is not None:
            fig.colorbar(points,ax=(ax_histx,ax,ax_histy))
        fig.suptitle(title,fontsize=title_fontsize,x=title_x,y=title_y)
        return fig

class PCA_3D:
    def __init__(self, data: np.array):
        self.pca = PCA(n_components=3)
        self.init_fit(data)
        pass

    def init_fit(self, data: np.array):
        self.pca.fit(data)
        print('Explained variance: ', self.pca.explained_variance_ratio_)
    
    @staticmethod
    def plot_proj(x:np.array, y:np.array, z:np.array, c:np.array = None, cmap:str = 'crest', s=10, 
                    figsize:tuple = (8, 8), title:str = '', title_fontsize = 14, title_x = 0.45, title_y = 1.0, 
                    x_label = 'x', y_label = 'y', z_label = 'z',
                    surface = False, view = None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                    bbox_to_anchor = (1.03, 0., 1, 1), bbox_transform = ax.transAxes,
                    borderpad = 2)
        if c is not None:
            cmap = sns.color_palette(cmap, as_cmap=True)
            points = ax.scatter(x, y, z, c=c, s=s, cmap=cmap)
        else:
            points = ax.scatter(x, y, z, s=s)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        if c is not None:
            fig.colorbar(points,cax = axins)
        if surface:
            ax.plot_trisurf(x, y, z, cmap='crest', edgecolor='none')
        if view is not None:
            ax.view_init(elev=view[0], azim=view[1])
        fig.suptitle(title,fontsize=title_fontsize,x=title_x,y=title_y)
        return fig

    def plot(self, data: np.array, c:np.array = None, cmap:str = 'crest', s=10, 
                figsize:tuple = (8, 8), title:str = '', title_fontsize = 14, title_x = 0.45, title_y = 1.0, 
                x_label = 'x', y_label = 'y', z_label = 'z', 
                verbose = True):
        print('--- PCA 3D PLOT ---') if verbose else None
        print('Data shape: ', data.shape) if verbose else None
        pca_proj = self.pca.transform(data)
        pca_df = pd.DataFrame(pca_proj,columns=['x','y','z'])
        if c is not None:
            print('Coloring enabled.') if verbose else None
            print('Color shape: ', c.shape) if verbose else None
        
        print('Plotting...') if verbose else None
        fig = self.plot_proj(pca_df.x, pca_df.y, pca_df.z, c=c, cmap=cmap, s=s, 
                                figsize=figsize, title=title, title_fontsize=title_fontsize, title_x=title_x, title_y=title_y,
                                x_label=x_label, y_label=y_label, z_label=z_label)
        return fig