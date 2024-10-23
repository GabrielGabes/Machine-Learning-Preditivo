from sklearn.datasets import make_classification
import pandas as pd

# Gerar dados sintéticos desbalanceados
x, y = make_classification(n_samples=7000, 
                           n_features=10, # qtd total de features
                           n_informative=6, # features realmente uteis para o modelo
                           n_redundant=3, # features que diz praticamente oque uma outra já fim
                           n_classes=2, 
                           n_clusters_per_class=3,  # Subgrupos
                           weights=[0.93, 0.07], 
                           class_sep=0.2, # separação entre as classes
                           random_state=3141592)

# Criar um DataFrame
df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
df['target'] = y