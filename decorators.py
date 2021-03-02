from cluster import clustering_decorator, clustering
import data

masters_data=data.read_csv("datasets/masters.csv")
a=clustering_decorator(clustering)(masters_data)
print(a)