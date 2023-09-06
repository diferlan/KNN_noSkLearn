#Lo primero que se realiza es importar las liberias necesarias. Se usa sklearn para la division de datos, evaluacion del modelol y 
#preprocesamiento de los datos

#Liberia usada para manejar el conjunto de datos
import pandas as pd 
#Liberia usada para operaciones matematicas y manejo de arreglos
import numpy as np 
#Función para la division de datos en conjunto de prueba y entrenamiento 
from sklearn.model_selection import train_test_split
#Funciones usadas para la evaluacion del desempeno en la clasificacion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#Función para escalar los datos y su valores se ubiquen entre 0 y 1, parte del preprocesamiento de los datos
from sklearn.preprocessing import MinMaxScaler 
#Nos permite usar la funcion de sleep para dejar tiempo entre los prints y no perder informacion
import time 
#Usada para graficar los resultados
import matplotlib.pyplot as plt 

accuracys,recalls,precisions,f1s = [],[],[],[] #Inicializa los arreglos donde se guardaran los resultados
print('Inicializando algoritmo KNN hecho a mano:')
time.sleep(1) #Usamos sleep para espaciar ligeramente los print y se puedan leer
path = "stroke_data.csv" #Nombre del archivo de datos
print(f'Asegurese que es encuentra en la misma carpeta que el archivo {path}') #Pequeño mensaje, la idea es que se lea si el programa da error por no encontrar el archivo
time.sleep(1)
stroke = pd.read_csv(path) #Lectura del archivo csv usando pandas
#print(f'Se usan los datos de {path}\nLas columnas que contiene son: {stroke.columns}')
#time.sleep(1)
#Selección de datos para hacer más ligero computacionalmente el programa 
#Ya que el programa calcula la distancia de los datos de 'prueba' con TODOS los datos de entrenamiento, 
#el crecimiento del tiempo de entrenamiento es exponencial, se usa una muestra de los datos para mostrar el funcionamiento 
stroke = pd.concat([stroke.head(1000),stroke.tail(1000)])

#Eliminar los valores nulos
stroke.dropna(subset=['sex'],inplace=True)

#Se dividen los datos en x,y
x = stroke.drop('stroke',axis=1) #x contiene a las variables
y = stroke['stroke'] #y corresponde a las etiquetas

#Se escalan las datos, esto para evitar que los resultados se distorcionen
escalar = MinMaxScaler() #Creación del objeto de escalamiento
x = escalar.fit_transform(x) #escalamiento

#Función clave del knn, usa la norma 2 para calcular la distancia entre los puntos. 
# Recibe 2 puntos y saca la norma 2
def distance(x1,x2):
  #Operamos con arreglos, usamos la formula que se encuentra debajo
  #La resta de arreglos se realiza elemento con elemento, igualmente suscede esto con el cuadrado
  #Con np.sum imitamos la sumatoria en matematicas y obtenemos un resultado unico y sacamos la raiz cuadrada
  return np.sqrt(np.sum((x2 - x1)**2)) 

#Imprimimos un pequeño mensaje para que el usuario sepa que se esta haciendo algo y no se trabo el programa
print('Entrenando...')
time.sleep(2)

#Esta funcion contiene el algoritmo. El algoritmo knn recibe un punto y ubica el punto, despues identifica los k puntos con menor distancia 
# y revisa la categoria a la que pertenecen esos puntos, el algoritmo clasifica el nuevo punto de acuerdo a la categoria que más repita
#Por esto, se utiliza generalmente un numero impar para k, para evitar tener el mismo numero de ocurrencias en varias categorias
def knn(X_train,X_test, y_train,k=3):
  print(f'\n\nEntrenando con k={k}, espere...')
  y_pred = [] #Array que guardará las predicciones
  for x in X_test: #Itera en todos los valores del conjunto de prueba
    #Calcula las distancias de un punto de la prueba con cada punto del conjunto de entrenamiento
    distances= np.array([distance(x,x_sample) for x_sample in X_train]) #Se usa list comprenhension para iterar y sacar las distancias
    nn_labels = np.argsort(distances)[:k] #Toma los indices de los k puntos más cercanos. np.argsort devuelve los indices de los valores más pequeño de un arreglo. Usando slicers tomamos los k valores más pequeños
    labels = y_train.iloc[nn_labels] #Selecciona los labels de los índices recuperados en el paso anterior
    nn_group = np.bincount(labels).argmax() #Cuenta las ocurrencias en la selección de labels y toma regresa el label que más se repite
    y_pred.append(nn_group) #Agrega la predicción al arreglo de predicciones
  return np.array(y_pred) #Regresa el arreglo de predicciones para la evaluación posterior

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=19,shuffle=True) #Division de conjuntos de entrenamiento y prueba. Usamos random_state para obtener valores consistentes volviendo a entrenar el modelo
for k in range(1,12,2): #Usamos un ciclo con diferente valoress de k para explorar que valor de k es mas adecuado
  y_pred = knn(X_train,X_test,y_train,k) #Corre el algoritmo knn que se diseño
  print(f'Resultados con {k}nn:')
  #print(pd.DataFrame({'Reales': y_test,'Predicciones':y_pred }))
  time.sleep(1)
  #Una vez con los resultados, el siguiente paso es la evaluación del modelo
  #Usamos 4 métricas para observar el comportamiento,  anteriormente importamos la funciones para la evaluación
  accuracy,recall,precision,f1 = accuracy_score(y_test, y_pred), recall_score(y_test,y_pred) , precision_score(y_test,y_pred), f1_score(y_test,y_pred)
  #Al principio del código, inicializamos los arreglos para las metricas, guardamos las metrcias en cada iteración en su arreglo correspondiente
  accuracys.append(accuracy), recalls.append(recall), precisions.append(precision), f1s.append(f1)
  #Imprimimos las metricas
  print(f'Accuracy: {accuracy}')
  print(f'Recall: {recall}')
  print(f'Precision: {precision}')
  print(f'F1 Score: {f1}')
  print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
  #Finalmente, imprimos la cantidad de predicciones correctas en comparación con el total
  print(f'{np.sum(y_test==y_pred)} predecidos correctamente de {len(y_test)}')

#Resultados
#Con el fin de visualizar mejor los resultados gráficamos las matercias con cada k usada
ks = np.array(range(1,12,2)) #Crea el arreglo con las ks usadas
fig, axs = plt.subplots(1,4, figsize=(24,6)) #Creamos la figura a graficar

colors = ['#d4dbea' if value != max(accuracys) else '#536fab' for value in accuracys] #Usamos list comprenhension para colorear las barras, el máximo valor tendrá valor diferente
axs[0].bar(ks,accuracys,color=colors) #Creamos la grafica
axs[0].set_title('Accuracy')  #Ponemos el titulo correspondiente 
axs[0].text(6.7, 0.75, f'Máximo:{max(accuracys):.4f}', fontsize=12) #Agregamos un texto aclarando cual es el valor máximo y que se explique mejor la gráfica

#Se repite este proceso para las otras 3 métricas
colors = ['#d4dbea' if value != max(recalls) else '#536fab' for value in recalls]
axs[1].bar(ks,recalls,color=colors)
axs[1].set_title('Recall')
axs[1].text(6.7, 0.75, f'Máximo:{max(recalls):.4f}', fontsize=12)

colors = ['#d4dbea' if value != max(precisions) else '#536fab' for value in precisions]
axs[2].bar(ks,precisions,color=colors)
axs[2].set_title('Precision')
axs[2].text(6.7, 0.75, f'Máximo: {max(precisions):.4f}', fontsize=12)

colors = ['#d4dbea' if value != max(f1s) else '#536fab' for value in f1s]
axs[3].bar(ks,f1s,color=colors)
axs[3].set_title('F1')
axs[3].text(6.7, 0.75, f'Máximo: {max(f1s):.4f}', fontsize=12)

#En un ciclo agregamos los elementos que son comunes en las graficas.
for ax in axs:
  ax.set_xticks(ks) #el valor de las marcas en x se pone como las ks usadas
  ax.set_xlabel('K Vecinos') #Aclaramos en el eje X que se refiere a los K Vecinos
  ax.set_ylim([0,.80]) #Ponemos el límite de la grafica para que todas tengan el mismo límite y se sean congruentes

plt.tight_layout() #Evita que se encimen las graficar
plt.show() #Mostramos la imagen

print(f'''
      Por la naturaleza del problema, en este caso la métrica más importante sería recall. Nos interesa medir la proporción de casos positivos clasficados
      correctamente y minimizar la detección de falsos negativos para empezar tratamientos preventivos o vigilancia sobre el paciente. El máximo valor 
      obtenido para recall fue {max(recalls)} con {ks[np.argsort(recalls)[-1:]]}-vecinos más cercanos.

      ''')
"""##Comparación de rendimiento contra el KNN implementado con SKLearn"""
'''
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
'''
