# базовые операции с матрицами и датасетами
import numpy as np
import pandas as pd

# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hists_and_boxes(df, columns):
    """
    Построение боксплотов и гистограмм для списка колонок датафрейма
    """
    for col in columns:
        if df[col].dtype == object:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(col)
        sns.boxplot(df[col], ax=ax1)
        sns.distplot(df[col], ax=ax2)

def plot_distribution(df, var, target, **kwargs):
    '''
    Построение распределения плотносит ядра'''
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim = (df[var].min(), df[var].max()))
    facet.add_legend()

def plot_tree_feat_importances(data, tree):
    """
    Визуализация важности признаков для алгоритма случайный лес
    """
    feats = {}
    for feature, importance in zip(data.columns, tree.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight = 'bold')
    plt.ylabel('Features', fontsize=25, weight = 'bold')
    plt.title('Feature Importance', fontsize=25, weight = 'bold')
    display(plt.show())