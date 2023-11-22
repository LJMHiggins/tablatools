## Data inspecting functions ##
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#split training qualitative and quantitative data for EDA
def retrieve_quantitative_qualitative(df, target_col = None):
    quant = pd.DataFrame()
    qual = pd.DataFrame()
    for col in df.columns:
        if df[col].dtypes == 'object':
            qual[col] = df[col]
        else:
            quant[col] = df[col]
    if target_col is not None:
        qual[target_col] = df[target_col]
        quant[target_col] = df[target_col]
    return(quant, qual)

def normal_dist_hist(data, ax = None):

    if data.isnull().values.any() == True:
        data.dropna(inplace=True)
    if data.dtypes == 'float64':
        data.astype('int64')
    #Plot distribution with Gaussian overlay
    mu, std = stats.norm.fit(data)
    if ax is not None:
        ax.hist(data, bins=50, density=True, alpha=0.6, color='g')
    else:
        fig, ax = plt.subplots()
        ax.hist(data, bins=50, density=True, alpha=0.6, color='g')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    col_title = data.name
    title = "Data: " + col_title + " --  Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    ax.set_title(title)
    
def normal_test_QQplots(data, ax = None):
    #added ax argument for specifying subplot coordinate
    data.dropna(inplace=True)
    if ax is not None:
        probplt = stats.probplot(data,dist='norm',fit=True,plot=ax)
    else:
        fig, ax = plt.subplots()
        probplt = stats.probplot(data,dist='norm',fit=True, plot=ax)

def normality_report(df, figsize = None):
    if figsize is None:
        figsize = (20,100)
    fig, axes = plt.subplots(nrows=len(df.columns[1:]), ncols=2, figsize=figsize)
    ax_y = 0
    col_i = 1
    for col in df.columns[1:]:
        ax_x = 0
        normal_dist_hist(df[col], ax=axes[ax_y, ax_x])
        ax_x = 1
        normal_test_QQplots(df[col], ax=axes[ax_y, ax_x])
        ax_y += 1
        plt.show
    plt.tight_layout()
    
    #Visualise missing values
def visualise_missing_bar(df):
    miss = (df
            .isnull()
            .sum()
            .sort_values()
            )
    miss = miss[miss>0]
    if miss.empty:
        return print('Dataframe is empty')
    miss.plot.bar()

def missing_heatmap(df):
    plt.figure(figsize=(16,10))
    hm = sns.heatmap(data= df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 15)
    hm.set_title('Missing values', fontsize = 17)
    
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
    
def box_plots_for_target(df, target_col_name):
    long_df = pd.DataFrame()
    if df.isnull().values.any() == True:
        df.dropna(inplace=True)
    for col in df.columns:
        if df[col].dtypes == 'object' or col == target_col_name:
            long_df[col] = df[col]
    long_df = pd.melt(long_df, id_vars=target_col_name, 
                      value_vars = [col for col in long_df.columns if col != target_col_name])
    return long_df.head()