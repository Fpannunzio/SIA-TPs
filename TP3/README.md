# Redes Neuronales Basadas en Perceptrones

Librería de redes neuronales basadas en perceptrones. Soporta redes neuronales de un solo perceptron (step, lineal y no-lineal), y redes neuronales multicapa que propagan el error mediante [backpropagation](https://en.wikipedia.org/wiki/Backpropagation).

El método de aprendizaje utilizado se basa en una búsqueda lineal iterativa (no batch) mediante la estrategia de minimización del [gradiente descendiente](https://en.wikipedia.org/wiki/Gradient_descent).

## Autores

- [Brandy, Tobias](https://github.com/tobiasbrandy)
- [Pannunzio, Faustino](https://github.com/Fpannunzio)
- [Sagues, Ignacio](https://github.com/isagues)

## Dependencias

- Python 3.6+
- El resto de las dependencias se encuentran especificadas en `requirements.txt`

Contando con python3 instalado se pueden instalar las dependencias restantes con `pip install -r requirements.txt`

## Ejecución

Si bien el principal objectivo del proyecto es generar una pequeña librería interoperable en distintos contextos, se incluyen tres archivos de ejemplo de uso: `ej1.py`, `ej2.py` y `ej3.py`.

Para ejecutarlos no es necesario hacer ningún tipo de compilación, basta con ejecutarlo.

Para esto se debera correr, por ejemplo, `python ej2.py`. 

A menos que se indique lo contrario, se buscará en la raíz del proyecto el archivo `config.yaml` donde deberá estar toda la información de [configuracion](#configuracion). 

En caso de querer almacenar el archivo de configuración en otra ubicación o con otro nombre se debera proveer el path del mismo como argumento de ejecucion `python ej1.py <path>`.

Una vez ejecutado, se imprimirá a salida estándar el resultado del programa.

Además, se abrirán de forma automática gráficos (realizados en [matplotlib](https://matplotlib.org/)) mostrando información pertinente, según corresponda.

Dentro de las carpetas `trainingset/input` y `trainingset/output` se encuentran algunos datasets de prueba. Estos son los datasets default de los archivos de ejemplo provistos.

### Configuracion

La configuración de ejecución es realizada via un archivo de extensión [YAML](https://yaml.org/) el cual debera contener la siguiente información:

- `training_set:` *Opcional*
  - Configuración respecto al dataset de entrenamiento a utilizar
  - Si no se especifica, los ejemplos respetaran una configuración default. De especificarse, debe respetarse el siguiente formato:
  - `inputs: <input_dataset_file_path>`
    - El path a un archivo .tsv con los valores de los puntos de entrenamiento. Estos son desde la raiz del proyecto.
  - `input_line_count: <int>` *Opcional*
    - La cantidad de lineas en el dataset provisto que representa 1 punto.
    - Por ejemplo, en el dataset `trainingset/inputs/Ej3.tsv` cada punto esta representado como una matrix de bits de 7 filas x 5 columnas. Esto quiere decir que 7 filas corresponsen a un mismo punto. Para indicar esto, se debe setear `input_line_count` en 7.
    - Debe ser un entero positivo. Su valor default es 1.
  - `outputs: <output_dataset_file_path>`
    - Equivalente a `inputs`, pero para los valores de salida de los puntos ingresados por en `inputs`, en el mismo orden.
  - `output_line_count: <int>` *Opcional*
    - Equivalente a `input_line_count`, pero para los valores de salida.
  - `normalize_values: <bool>` *Opcional*
    - Este parámetro indica si se quiere normalizar los valores de input y output de [-100, 100] a [-1, 1].
    - Esto puede ser práctico para poder utilizar funciones sigmoideas en problemas de regresión.
    - Este parámetro está implementado como una division por 100 de todos los valores. Si se considera que este método no es adecuado para el dataset seleccionado, simplemente puede ignorarse este parámetro.
    - Su valor default es false.
    
- `plot: <bool>` *Opcional*
  - Setear en false si no se desea que se realize ningun grafico.
  - Por default en true.
    
- `network`
  - Configuración para especificar la red neuronal. Primero se indican los parámetros necesarios para elegir el tipo de red, junto con sus parámetros específicos. Luego, el resto de los parámetros son todos opcionales y aplican para cualquier tipo de red neuronal.
  - Los distintos tipos de redes neuronales disponibles son:
    - **Simple**
      - Red neuronal de un solo perceptron con la función de activación `sign` y la función de error `sum(abs(real_values - predicted_values))`.
      - Esta red solo es util para problemas de clasificación linealmente separables.
      - `type: simple`
    - **Linear**
      - Red neuronal de un solo perceptron cuya función de activación es la función identidad.
      - Red util para el cálculo de regresiones lineales
      - `type: linear`
      - `network_params: `
        - `error_function: <str>`
          - Parámetro donde se especifica la función de error a utilizar. 
          - Los posibles valores son:
            - `quadratic`
              - Muy similar al error cuadrático medio ([mse](https://en.wikipedia.org/wiki/Mean_squared_error)), pero siempre dividido por 2.
              - Definido como `sum((real_values - predicted_values) ** 2) / 2`
              - Es usado generalmente en problemas de aproximación.
              - Posee el problema de que cuanto más cerca del objetivo se está, menor será el nuevo delta_w.
              - Valor default
            - `logarithmic`
              - Definido como `sum((1 + real_values)*ln((1 + real_values)/(1 + predicted_values)) + (1 - real_values)*ln((1 - real_values)/(1 - predicted_values)))/2`.
              - Cuanto más cerca del objetivo se está, mayor será el delta_w.
              - En general usado en problemas de clasificación.
    - **No Lineal**
      - Red neuronal de un solo perceptron que soporta cualquier tipo de función de activación derivable, por lo que puede resolver problemas más complejos.
      - En general es utilizado con funciones sigmoideas.
      - `type: non_linear`
      - `network_params: `
        - `activation_function: <str>`
          - Función de activación del perceptron de la red.
          - Valores soportados:
            - `tanh`
              - [Tangente Hiperbólica](https://en.wikipedia.org/wiki/Hyperbolic_functions#Definitions)
              - `f(x) = tanh(bx)` donde b es el `activation_slope_factor`
            - `logistic`
              - [Función Logística](https://en.wikipedia.org/wiki/Logistic_function)
              - `f(x) = 1/(1 + exp(-2*b*x))` donde b es el `activation_slope_factor`
        - `activation_slope_factor: <float>`
          - Parámetro usado en el cálculo de la función de activación, cuando se elige una función sigmoidea.
          - Parametriza la inclinación de la función.
          - Número decimal positivo, por lo general entre 0 y 1.
        - `error_function: <str>`
          - Este parámetro es equivalente al parámetro del mismo nombre en la red neuronal **Lineal**.
    - **Multicapa**
      - Red neuronal compuesta por multiples capas con distintas cantidades de perceptrones no lineales.
      - Este tipo de red permite resolver problemas linearmente no separables.
      - Para el aprendizaje utiliza el método de backpropagation.
      - `type: multi_layered`
      - `network_params: `
        - `layer_sizes: <list[int]>`
          - Especifica la configuración de capas de la red. Cada entero positivo del array configura la cantidad de perceptrones de la capa.
          - La capa de la izquierda es la de entrada y la derecha la de salida. Las intermedias son capas ocultas.
          - El tamaño de la capa de entrada debe ser la misma que la dimension de los puntos de entrada. El tamaño de la capa de salida debe ser la misma que la dimension de los valores de los puntos.
          - La lista debe tener al menos tamaño 2, y solo se permiten enteros positivos.
        - `activation_function: <str>`
          - Este parámetro es equivalente al parámetro del mismo nombre en la red neuronal **No Lineal**.
        - `activation_slope_factor: <float>`
          - Este parámetro es equivalente al parámetro del mismo nombre en la red neuronal **No Lineal**.
        - `error_function: <str>`
          - Este parámetro es equivalente al parámetro del mismo nombre en la red neuronal **Lineal**.
      
  - `max_training_iterations: <int>` *Opcional*
    - Cantidad máxima de iteraciones de entrenamiento.
    - Debe ser un entero positivo. Por default 100.
  - `weight_reset_threshold`: <int> *Opcional*
    - Cada cuantas iteraciones se empieza el aprendizaje de nuevo, randomizando el w actual.
    - Debe ser un entero positivo. Si es mayor a `max_training_iterations`, no tiene efecto. Por default, no se realiza nunca.
  - `max_stale_error_iterations: <int>` *Opcional*
    - Cortar ejecucion cuando el error no cambia por esta cantidad de iteraciones.
    - Debe ser un entero positivo. Por default, nunca corta ejecución por esta razón.
  - `error_goal: <float>` *Opcional*
    - El error maximo que se desea alcanzar. Si se consigue que el error sea menor a este valor, se corta la ejecucion.
    - Debe ser un valor decimal positivo. Por default es 0.
  - `error_tolerance: <float>` *Opcional*
    - Epsilon utilizado en todos los cálculos de punto flotante relacionados con el error.
    - Debe ser un numero decimal positivo. Por default, 1e-9.
  - `momentum_factor: <float>` *Opcional*
    - El factor que se multiplica el delta w anterior antes de sumarlo al actual en la actualización del w.
    - El momentum esta implementado como: `w = w + nuevo_delta_w + momentum_factor*viejo_delta_w`.
    - Debe ser un decimal positivo. Por default, su valor es 0, es decir, no se aplica momentum.
  - `learning_rate_strategy: <str>` *Opcional*
    - Indica la estrategia a usar para el calculo del learning rate.
    - Dependiendo de su valor, se habilitan nuevos parametros.
    - El valor default es `fixed`
    - Las posibles estrategias son son:
      - **Fixed**
        - Se configura un learning rate fijo que no se modifica durante todo el entrenamiento.
        - Si se elige un buen learning rate puede funcionar bien, pero es facil elegir uno muy grande o chico.
        - `learning_rate_strategy: fixed`
        - `base_learning_rate: <float>`
          - El learning rate a utilizar durante todo el entrenamiento.
      - **Variable**
        - Se establece un learning rate inicial.
        - Si durante una cierta cantidad de iteraciones el error disminuye, se empieza a aumentar el learning rate de a pasos constantes.
        - Si durante una cierta cantidad de iteraciones el error aumenta, se empieza a disminuir el learning rate por un porcentaje de su valor (cuanto mas grande, mas rapido disminuye).
        - Resulta bueno para ir ajustando el learning rate a un costo computacional muy bajo.
        - `learning_rate_strategy: variable`
        - `base_learning_rate: <float>`
          - El learning rate inicial.
        - `variable_learning_rate_params: `
          - `up_scaling_factor: <float>` *Opcional*
            - Valor fijo por el que se aumenta el learning rate.
            - Debe ser un decimal positivo. Por default, es 0.
          - `down_scaling_factor: <float>` *Opcional*
            - Porcentaje por el que se disminuye el learning rate.
            - Debe ser un decimal positivo. Por default, es 0.
          - `positive_trend_threshold: <int>` *Opcional*
            - La cantidad de iteraciones consecutivas que el error debe disminuir para que se empiece a aumentar el learning rate.
            - Debe ser un entero positivo. Por default, es infinito.
          - `negative_trend_threshold: <int>` *Opcional*
            - La cantidad de iteraciones consecutivas que el error debe aumentar para que se empiece a disminuir el learning rate.
            - Debe ser un entero positivo. Por default, es infinito.
      - **Búsqueda Lineal**
        - Busca el learning rate óptimo para cada iteración mediante búsqueda lineal con el método del gradiente descendiente.
        - De esta manera, no es necesario adivinar ningún learning rate, sino que se intentara buscar el mejor.
        - `learning_rate_strategy: linear_search`
        - `learning_rate_linear_search_params: `
          - `max_iterations: <int>`
            - Máxima cantidad de iteraciones del método de búsqueda lineal. Se queda con el mejor que encontró.
            - Debe ser un entero positivo. Por default, es 500.
          - `error_tolerance: <float>`
            - La tolerancia de error a tener durante el método de búsqueda lineal.
            - Decimal positivo. Por default, 1e-5.
          - `max_value: <int>`
            - Valor maximo de learning rate buscado.
            - Entero positivo. Por default, 1.
Ejemplos de configuracion

```yaml
  TODO: PONER EJEMPLO 1
```
```yaml
    TODO: PONER EJEMPLO 2
```

### Resultado

`TODO:` 
