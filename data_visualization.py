import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import seaborn as sns
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from AnalysisMethods import tSNE, PCA, UMAP
from utils import CheckSavePath, GetNowTime_yyyymmddhhMMss

# methods_map = {'pca': PCA.PCA(), 'tsne': tSNE.TSNE(), 'umap': UMAP.UMAP()}

def ClassColors(classes= ["A", "B", "C", "D", "E"]):
    class_colors = dict()
    classes_count = len(classes)
    # plot_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'yellow', 'tomato']
    plot_colors = ['b', 'g', 'r', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    sample_colors = random.sample(plot_colors, classes_count)
    for sample_color, class_name in zip(sample_colors, classes):
        class_colors[class_name] = sample_color
    return class_colors

class DataVisualization:
    def __init__(self, analysis='pca', save_path='./analysis_images'):
        self.analysis = analysis
        self.save_path = CheckSavePath(save_path)
        self.save_img_name = ""

    def dataTransform(self, data_dict, gray_scale=False, norm=False):
        data_X = []
        data_Y = []
        for item in data_dict:
            image=None
            if gray_scale:
                image = cv2.cvtColor(item['image'], cv2.COLOR_BGR2GRAY)
            else:
                image = item['image']
            data_X.append(image.flatten())
            data_Y.append(item['label'])
        
        data_X = np.asarray(data_X)
        if norm:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data_X)
            data_X = scaler.transform(data_X)
        data_Y = np.asarray(data_Y)
        return data_X, data_Y
    
    def Draw2DImage(self, analysis_result_df, lim, x_name='x', y_name='y', title='Analysis Result'):
        
        # Plotting
        fig = plt.figure(figsize=(30,30))
        ax = fig.add_subplot(1, 1, 1)
        sns.scatterplot(x=x_name, y=y_name, hue='label', style='label', data=analysis_result_df, ax=ax,s=120)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.0)
        ax.set_title(title)
        self.save_img_name = "{}_{}.png".format(GetNowTime_yyyymmddhhMMss(), self.analysis)
        save_image_path = os.path.join(self.save_path, self.save_img_name)
        plt.savefig(save_image_path)
        plt.show()
    
    def Draw3DImage(self, analysis_result_df, lim, x_name='x', y_name='y', z_name='z', title='Analysis Result'):
        
        fig = plt.figure(figsize=(30,30))
        ax = fig.add_subplot(projection='3d')
        class_list = np.unique(analysis_result_df['label'])
        class_colors = ClassColors(class_list.tolist())
        for name in class_list.tolist():
            lable_df = analysis_result_df[analysis_result_df['label']==name]
            ax.scatter(lable_df[x_name], lable_df[y_name], lable_df[z_name], alpha=0.8, c=class_colors[name], edgecolors='none', s=40, label=name)
        # for x, y, z, group in zip(analysis_result_df[x_name].values, analysis_result_df[y_name].values, analysis_result_df[z_name].values, analysis_result_df['label']):
        #     ax.scatter([x], [y], [z], alpha=0.8, c=class_colors[group], edgecolors='none', s=40, label=group)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(z_name)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_title(title)
        ax.set_aspect('auto')
        ax.legend(class_list, bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.0)
        self.save_img_name = "{}_{}.png".format(GetNowTime_yyyymmddhhMMss(), self.analysis)
        save_image_path = os.path.join(self.save_path, self.save_img_name)
        plt.savefig(save_image_path, bbox_inches='tight')
        plt.show()
    
    def visualization(self, data_X, data_Y, sampled_data=1000):
        """Using PCA or tSNE for data visualization.
        Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - analysis: tsne or pca
        """
        # Analysis sample size (for faster computation)
        subsample_idc = np.random.choice(data_X.shape[0], sampled_data, replace=True)
        data_x = data_X[subsample_idc,:]
        # y = np.unique(data_Y)
        data_y = data_Y[subsample_idc]

        analysis_result = None
        lim = None
        analysis_result_df = None
        
        if self.analysis == 'pca':
            pca = PCA.PCA()
            analysis_result = pca(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)
            if analysis_result.shape[1] == 2:
                analysis_result_df = pd.DataFrame({'x-pca': analysis_result[:,0], 'y-pca': analysis_result[:,1], 'label': data_y})
                self.Draw2DImage(analysis_result_df, lim, x_name='x-pca', y_name='y-pca', title='PCA plot')
            if analysis_result.shape[1] == 3:
                analysis_result_df = pd.DataFrame({'x-pca': analysis_result[:,0], 'y-pca': analysis_result[:,1], 'z-pca': analysis_result[:, 2], 'label': data_y})
                self.Draw3DImage(analysis_result_df, lim, x_name='x-pca', y_name='y-pca', z_name='z-pca', title='PCA 3D plot')

        elif self.analysis == 'tsne':
            tsne = tSNE.tSNE()
            analysis_result = tsne(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)
            if analysis_result.shape[1] == 2:
                analysis_result_df = pd.DataFrame({'x-tsne': analysis_result[:,0], 'y-tsne': analysis_result[:,1], 'label': data_y})
                self.Draw2DImage(analysis_result_df, lim, x_name='x-tsne', y_name='y-tsne', title='tSNE plot')
            if analysis_result.shape[1] == 3:
                analysis_result_df = pd.DataFrame({'x-tsne': analysis_result[:,0], 'y-tsne': analysis_result[:,1], 'z-tsne': analysis_result[:, 2], 'label': data_y})
                self.Draw3DImage(analysis_result_df, lim, x_name='x-tsne', y_name='y-tsne', z_name='z-tsne', title='tSNE 3D plot')
        
        elif self.analysis == 'umap':
            umap = UMAP.UMAP()
            analysis_result = umap(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)
            if analysis_result.shape[1] == 2:
                analysis_result_df = pd.DataFrame({'x-umap': analysis_result[:,0], 'y-umap': analysis_result[:,1], 'label': data_y})
                self.Draw2DImage(analysis_result_df, lim, x_name='x-umap', y_name='y-umap', title='UMAP plot')
            if analysis_result.shape[1] == 3:
                analysis_result_df = pd.DataFrame({'x-umap': analysis_result[:,0], 'y-umap': analysis_result[:,1], 'z-umap': analysis_result[:, 2], 'label': data_y})
                self.Draw3DImage(analysis_result_df, lim, x_name='x-umap', y_name='y-umap', z_name='z-umap', title='UMAP 3D plot')


