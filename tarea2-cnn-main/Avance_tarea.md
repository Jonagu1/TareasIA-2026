# Resumen de Avance - Tarea 2 

## 1. El Desbalance de Clases y la Fuga de Datos (Data Leakage) 
Como el profe avisó que el desbalance era el desafío principal, y ya no estamos usando la aumentación en esta etapa, lo resolví matemáticamente:
* **Pesos en la Función de Pérdida:** Calculé cuántos Pokémon hay de cada tipo en el entrenamiento y le pasé esos "pesos" al `CrossEntropyLoss`. Ahora, si la red se equivoca en un Pokémon raro, el castigo numérico es mucho mayor.
* **Normalización Estricta:** Estandaricé los atributos numéricos (Altura, Peso, HP, etc.) para que la red Densa no se vuelva loca con las escalas. Lo importante aquí es que calculé la media y la desviación estándar **solo con los datos de entrenamiento** y usé esos mismos números para evaluar el test, evitando así la temida "Fuga de Datos" (hacer trampa).

## 2. Modelo Finalizados 
El modelo base multiclase (CNN + MLP fusionados) ya corrió por 30 épocas. La curva de aprendizaje bajó súper bien.


##3 Actualización del modelo , evaluacion y conteo de errores
Se grafica la curva de aprendizaje : Loss / Epocas
Para evaluarlo se imprime precision , recall , f1_score, y suppor (la cantidad de datos de validación)
Luego se grafica la matriz de confusion para ver como rindió el modelo ante cada tipo de pokemon
El analisis de errores basicamente toma la lista de valores reales y predictos y cuenta las diferencias, se imprime este número y se muestra un par de ejemplos.

##4 Aumentacion de datos. Se hace una aumentacion en linea del train dataset
Se aumentan los datos con rotaciones, cambios de brillo , cambio de tamaño.
Luego se hacen todas las pruebas anteriores para ver el nuevo rendimiento.

