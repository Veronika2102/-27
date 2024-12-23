# Импорт необходимых библиотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from matplotlib.gridspec import GridSpec
# Загрузка датасета Iris
iris = sns.load_dataset('iris')
iris
# Визуализация с Matplotlib
# Гистограмма
plt.figure(figsize=(5, 5))
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.hist(subset['sepal_length'], alpha=0.5, bins=5, label=species)
plt.title('Гистограмма длины чашелистика (Sepal Length)  c помощью Matplotlib')
plt.xlabel('Длина чашелистика')
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.show()
# Диаграмма рассеивания
fig = plt.figure()
gs = GridSpec(4, 4)
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[1, 0:3])
ax_hist_y = fig.add_subplot(gs[1:4, 3])
ax_scatter.scatter(iris['sepal_length'], iris['sepal_width'])
ax_hist_x.hist(iris['sepal_length'])
ax_hist_y.hist(iris['sepal_width'], orientation='horizontal')
plt.show()
# Парный график
plt.figure(figsize=(12,6), dpi= 100)
sns.pairplot(iris, kind="scatter", hue="species", plot_kws=dict(s=100, edgecolor="white", linewidth=3.5))
plt.show()
# Круговая диаграмма
species_counts = iris['species'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(species_counts,labels=species_counts.index, autopct='%2.2f%%', startangle=100)
plt.show()

# Везуализация с Seaborn
# Гистограмма
plt.figure(figsize=(5, 5))
sns.histplot(data=iris, x='sepal_length', hue='species',alpha=0.5, bins=5, kde=False)
plt.title('Гистограмма длины чашелистика (Sepal Length) с помощью Seaborn')
plt.xlabel('Длина чашелистика')
plt.ylabel('Частота')
plt.grid(True)
plt.show()
# Диаграмма рассеивания
sns.scatterplot(x='sepal_length',y='petal_length',hue='species',style='species',s=100,data=iris)
plt.show()
# Линейный график
sns.lineplot(x='petal_length',y='petal_width',data=iris)
plt.show() 
# График плотности
sns.kdeplot(x='petal_length',data=iris,hue='species',multiple='stack')
plt.show()

# Визуализация с Plotly
# Гистограмма
fig = px.histogram(iris, x='sepal_length', color='species', barmode='overlay',
                   title='Гистограмма длины чашелистика (Sepal Length) - Plotly',
                   labels={'sepal_length': 'Длина чашелистика', 'count': 'Частота'})
fig.update_layout(xaxis_title='Длина чашелистика', yaxis_title='Частота')
fig.show()
# Диаграмма рассеивания
fig = px.scatter(iris, x="sepal_length", y="sepal_width", color="species")
fig.show()
# Линейный график
fig = px.line(iris, y="sepal_width",)
fig.show()
# График скрипки
fig = px.violin(iris, x="sepal_length", y="sepal_width", color="species")
fig.show()

