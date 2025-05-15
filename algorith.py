import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Данные
brands = ['Toyota', 'Hyundai', 'Kia', 'Renault', 'Volkswagen', 'Ford', 'Skoda', 'Chevrolet', 'Nissan', 'BMW']
sales = [4000, 2500, 2000, 1500, 1200, 1000, 800, 700, 600, 500]

# Преобразуем в numpy
X = np.array(sales).reshape(-1, 1)

# Кластеризация
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_.flatten()

# Определим где больше продаж
high_cluster = np.argmax(centers)
colors = ['blue' if label == high_cluster else 'red' for label in labels]

# X координаты — просто индексы
x_coords = np.arange(len(brands))

# Построим scatter
plt.figure(figsize=(10, 6))
for i in range(len(brands)):
    plt.scatter(x_coords[i], sales[i], color=colors[i], s=100)
    plt.text(x_coords[i], sales[i]+50, brands[i], ha='center', fontsize=9)

# Настройки
plt.xticks([])
plt.title("Кластеризация продаж (точки)", fontsize=14)
plt.ylabel("Продажи")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Много продаж', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Мало продаж', markerfacecolor='red', markersize=10)
])

# Сохранить и показать
plt.tight_layout()
plt.savefig("car_sales_clusters_scatter.png")
plt.show()
