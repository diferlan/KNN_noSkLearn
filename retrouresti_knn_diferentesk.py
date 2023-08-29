
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #División de prueba
from sklearn.metrics import accuracy_score #Revisa el accuracy
import time


#path= '/content/drive/MyDrive/7 Semestre/Uresti/stroke_data.csv'
print('Inicializando algoritmo KNN hecho a mano:')
time.sleep(1)
path = "stroke_data.csv"
print(f'Asegurese que es encuentra en la misma carpeta que el archivo {path}')
time.sleep(1)
stroke = pd.read_csv(path)
print(f'Se usan los datos de {path}\nLas columnas que contiene son: {stroke.columns}')
time.sleep(1)
stroke.shape
#Selección de datos para hacer más ligero computacionalmente el programa
stroke = pd.concat([stroke.head(1000),stroke.tail(1000)])
#Eliminar los valores nulos
stroke.dropna(subset=['sex'],inplace=True)
#Chequea los tipos de datos, para usar KNN queremos solo númericos
#stroke.dtypes
stroke.head(2)
x = stroke.drop('stroke',axis=1)
y = stroke['stroke']
k = len(y.unique())

from sklearn.preprocessing import StandardScaler
#normalizar = StandardScaler()
#x= normalizar.fit_transform(x)
from sklearn.preprocessing import MinMaxScaler
escalar = MinMaxScaler()
x = escalar.fit_transform(x)

def distance(x1,x2):
  return np.sqrt(np.sum(x1 - x2)**2)
print('Entrenando...')
time.sleep(2)
def knn(X_train,X_test, y_train,k=3):
  y_pred = [] #Array que guardará las predicciones
  for x in X_test: #Itera en todos los valores
    #print('.') 
    #Calcula las distancias de un punto de la prueba con cada puntos
    distances= np.array([distance(x,x_sample) for x_sample in X_train])
    nn_labels = np.argsort(distances)[:k] #Toma los indices de los k puntos más cercanos
    labels = y_train.iloc[nn_labels] #Selecciona los labels de los índices
    nn_group = np.bincount(labels).argmax() #Cuenta las ocurrencias en la selección de labels y toma el valor máximo
    y_pred.append(nn_group) #Agrega la predicción al arreglo de preducciónes
  return np.array(y_pred)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=19,shuffle=True) #Divisón de conjuntos de entrenamiento y prueba
for k in range(1,12,2):
  y_pred = knn(X_train,X_test,y_train,k)
  print('Resultados con {k}nn:')
  print(pd.DataFrame({'Reales': y_test,'Predicciones':y_pred }))
  time.sleep(2)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Accuracy: {accuracy}')
  print(f'{np.sum(y_test==y_pred)} predecidos correctamente de {len(y_test)}')


"""##Comparación de rendimiento contra el KNN implementado con SKLearn"""
'''
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
'''
