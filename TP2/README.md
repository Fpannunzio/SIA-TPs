# RPG Character Optimizer

RPG Character Optimizer busca encontrar el mejor personaje de RPG

## Autores

- [Brandy, Tobias](https://github.com/tobiasbrandy)
- [Pannunzio, Faustino](https://github.com/Fpannunzio)
- [Sagues, Ignacio](https://github.com/isagues)

## Dependencias

- Python 3.6+
- El resto de las dependencias se encuentran especificadas en `requirements.txt`

Contando con python3 instalado se pueden instalar las dependencias restantes con `pip install -r requirements.txt`

## Ejecucion

Para ejecutar el programa no es necesario hacer ningun tipo de compilacion, basta con ejecutarlo.

Para esto se debera correr `rpg_character_optimizer.py`. 

A menos que se indique lo contrario, se buscara en la raiz del proyecto el archivo `config.yaml` donde debera estar toda la informacion de [configuracion](#configuracion). 

En caso de querer almacenar el archivo de configuracion en otro ubicacion o con otro nombre se debera proveer el path del mismo como argumento de ejecucion `python sokoban_solver.py <path>`.

Una vez ejecutado se imprimira a salida estandar el estado del programa y se abrira de forma automatica una ventana mostrando graficos en tiempo real (hechos con [matplotlib](https://matplotlib.org/)) que informan sobre las generaciones de individuos que se van creando.

### Configuracion

La configuracion de ejecucion es realizada via un archivo de tipo [YAML](https://yaml.org/) el cual debera contener la siguiente informacion.

- `class: <str>`
  - Indica la clase del personaje que se intenta optimizar
  - Valores posibles:
    - `warrior`
    - `archer`
    - `defender`
    - `rogue`
- `seed: (<int>|<str>)` *Opcional*
  - Seed que se utilizara durante la simulacion para la generacion de numeros aleatorios. Esto permite repetir simulaciones pasadas de manera deterministica. Si no es especificado, se utilizara el [unix time](https://en.wikipedia.org/wiki/Unix_time) actual. La seed usada por la simulacion es impresa a salida estandar antes y despues de la simulacion.
- `output_file: str` *Opcional*
  - Path relativo al archivo de output deseado.
  - De ser configurado, el output de la simulacion (numero de generaciones, seed, mejor personaje generado y el numero de generacion al cual pertenece) sera adjuntado a dicho archivo (y creado de ser necesario), ademas de ser impreso a salida estandar.
- `population_size: int`
  - Entero positivo menor a 10.000
  - Indica el tamaño total de las poblaciones generadas en cada generacion.
- `item_files: `
  - `weapon: <item_pool_file_path>`
  - `boots: <item_pool_file_path>`
  - `helmet: <item_pool_file_path>`
  - `gauntlets: <item_pool_file_path>`
  - `chest_piece: <item_pool_file_path>`
    - Todas las propiedades anteriores deben indicar el path relativo un archivo [tsv](https://en.wikipedia.org/wiki/Tab-separated_values) donde se encuentre la informacion de la pool de items de tipo correspondiente.
    - La primer columna debe ser el id del objeto en forma ascendente
    - Se deben encontrar los valores decimales para las siguientes propiedades con su header correspondiente (no importa el orden):
      - `Fu` (strength)
      - `Ag` (agility)
      - `Ex` (experience)
      - `Re` (endurance)
      - `Vi` (vitality)
- `parent_selection: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar la seleccion de padres sobre la poblacion
  - `parent_count: <int>`
    - Cantidad de padres a seleccionar de la poblacion.
    - Entero mayor a 1
  - `weight: <float>`
    - Peso que se le da al metodo de seleccion 1 (y por lo tanto al 2)
    - Decimal entre 0 y 1.
    - El metodo de seleccion 1 eligira `ceil(weight * parent_count)` padres, mientras que el metodo 2 eligira `floor((1 - weight) * parent_count)`.
  - `method1: <method>`
  - `method2: <method>`
      - `method1` y `method2` configuran los dos metodos de seleccion que se usaran para elegir los padres de la nueva generacion. El valor `<method>` es un mapa que varia segun el metodo seleccionado.
      - Los distintos metodos de seleccion soportados son:
        - **Elite**
          - `name: elite`
        - **Ruleta**
          - `name: roulette`
        - **Universal**
          - `name: universal`
        - **Ranking**
          - `name: ranking`
          - `params: `
            - `roulette_method: <str>`
              - Especifica el metodo de ruleta a usar utilizando la nueva funcion de fitness calculada a partir de la funcion de ranking.
              - Los posibles valores son:
                - `roulette`
                - `universal`
        - **Boltzmann**
          - `name: boltzmann`
          -`params: `
            - `roulette_method: <str>`
              - Equivalente a la propiedad del mismo nombre en **Ranking**, pero usando la nueva funcion de fitness definida con por **Boltzmann**
            - `initial_temp: (<float>|<int>)`
              - Temperatura inicial. Numero positivo.
            - `final_temp: (<float>|<int>)`
              - Temperatura final. Numero positivo. Debe ser menor a `initial_temp`
            - `convergence_factor: (<float>|<int>)`
              - Factor de convergencia. Numero positivo (idealmente entre 0 y 1). Cuanto mas grande sea, mas rapido la tempreatura desciende.
              - La temperatura esta dada por la funcion `T(t) = final_temp + (initial_temp - final_temp) * exp(-convergence_factor*t)`
        - **Torneos Deterministicos**
          - `name: deterministic_tournament`
          - `params: `
            - `tournament_amount: <int>`
              - Tamaño del conjunto torneo sobre el cual se elige el ganador.
              - Entero positivo
        - **Torneo Probabilistico**
          - `name: probabilistic_tournament`
          - `params: `
            - `tournament_probability: <float>`
              - Probabilidad de que en un torneo (que es entre dos individuos) se elija al de mayor fitness.
              - Se elige un numero al azar p entre [0, 1] de manera uniforme. Si p < `tournament_probability`, entonces se elije al de mayor fitness. Sino, se elije al de menor.
- `parent_coupling: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar el emparejamiento del conjunto de padres.
  - `couple_count: <int>` *Opcional*
    - La cantidad de parejas a formar. Entero positivo.
    - De no ser especificado, se formaran `floor(parent_count/2)` parejas
  - `method: <method>`
    - `method` configura el metodo de emparejamiento que se usara. El valor `<method>` es un mapa que varia segun el metodo seleccionado.
    - Los distintos metodos de emparejamiento soportados son:
      - **Emparejamiento aleatorio caotico**
        - Este metodo de emparejamiento simplemente elige dos padres al azar y los empareja. Repite el proceso hasta que haya elegido la cantidad de parejas deseadas.
          - `name: chaotic_random`
      - **Emparejamiento aleatorio equitativo**
        - Este metodo de emparejamiento divide el conjunto de padres en dos y va formando parejas hasta que los conjuntos se vacian, o se alcanzo la cantidad de padres deseada. Si el conjunto se vacio, se vuelve a partir el conjunto original, y se repite el proceso.
          - `name: chaotic_random`
- `crossover: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar la creacion de hijos a partir de los padres (la cruza).
  - A partir de cada pareja de padres se crearan exactamente 2 hijos. Por lo tanto, el conjunto de hijos sera tan grande como el conjunto de padres. 
    - `children_eq_parents_prob: <float>`
      - Probabilidad de que el conjunto de hijos sea igual al conjunto de padres, es decir, que la creacion de hijos este solamente determinada por el proceso de mutacion.
    - `method: <method>`
      - `method` configura el metodo de cruza que se usara. El valor `<method>` es un mapa que varia segun el metodo seleccionado.
      - Los distintos metodos de cruza soportados son:
        - **Cruce Un Punto**
          - `name: single_point`
        - **Cruce Dos Puntos**
          - `name: two_point`
        - **Anular**
          - `name: annular`
        - **Uniforme**
          - `name: uniform`
          - `params: `
            - `weigth: float` *Opcional*
              - La probabilidad de que un gen en particular vaya del padre 1 al hijo 1 (y del padre 2 al hijo 2). Sino, sucede lo contrario.
              - Si no se especifica, este valor es 0.5 (el valor recomendado).
- `mutation: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar la mutacion de hijos.
    - `mutation_probability: <float>`
      - Probabilidad de que se mute. Dependiendo del metodo elegido, esta probabilidad se usa de distintas maneras para determinar si hay que mutar o no (sea un gen, o el individuo completo).
    - `method: <method>`
      - `method` configura el metodo de mutacion que se usara. El valor `<method>` es un mapa que varia segun el metodo seleccionado.
      - Los distintos metodos de mutacion soportados son:
        - **De 1 gen**
          - `name: single_gen`
        - **Multigen Limitada**
          - `name: limited`
          - `params: `
            - `max_mutated_genes_count: <int>`
              - Cantidad de genes maxima a mutar. Entero positivo que no puede superar la cantidad de genes (actualmente 6).
        - **Multigen Uniforme**
          - `name: uniform`
        - **Completa**
          - `name: complete`
- `survivor_selection: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar el metodo de seleccion de individuos sobrevivientes de una generacion a la sigueinte.
  - Este mapa se configura de manera equivalente al mapa `parent_selection`.
- `recombination: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar el conjunto sobre el cual se aplicara el metodo de seleccion de sobrevivientes de una generacion a la siguente.
  - Este conjunto sera una combinacion entre la poblacion de la generacion anterior y el conjunto de hijos recien creado.
    - `method: <method>`
      - `method` configura el metodo con el que se decidira sobre que conjunto se seleccionan los sobrevivientes. El valor `<method>` es un mapa que varia segun el metodo seleccionado.
      - Los distintos metodos de decision soportados son:
        - **Fill-All**
          - `name: fill-all`
        - **Fill-Parent**
          - `name: fill-parent`
- `end_condition: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar la condicion de corte.
  - El valor del mapa varia segun el metodo seleccionado. Las distintas condiciones de corte soportados son:
    - **Por Tiempo**
      - Corta la ejecucion una vez que la simulacion haya corrido una cantidad de segundos determinada.
        - `name: by_time`
        - `params: `
          - `runtime: (<float>|<int>)`
            - Cantidad de segundos que debe correr la simulacion antes de cortar
    - **Por Cantidad de Generaciones**
      - Corta la ejecucion una vez que la simulacion haya llegado a un numero de generacion determinado.
        - `name: by_generation`
        -`params: `
          - `limit_generation: <int>`
            - Numero de generacion donde se debe cortar ejecucion.
            - Entero positivo menor o igual a 1000.
    - **Por Valor de Fitness**
      - Corta la ejecucion una vez que el mejor valor de fitness de la generacion supera un cierto valor.
        - `name: by_fitness`
        - `params: `
          - `target_fitness: (<float>|<int>)`
            - Valor de fitness positivo que indica cuando cortar ejecucion.
          - `limit_generation: <int>` *Opcional*
            - De no haberse alcanzado el valor `target_fitness` para la generacion numero `limit_fitness`, la ejecucion es interrumpida de todas maneras. Busca evitar casos de loops infinitos.
            - Entero positivo entre 1 y 1000. De no ser especificado, el valor default es 1000.
    - **Por Convergencia de Fitness**
      - Corta la ejecucion cuando el valor del mejor fitness de la generacion se mantiene dentro de un epsilon por una cantidad determinada de generaciones.
        - `name: by_fitness_convergence`
        - `params: `
          - `epsilon: <float>`
            - Intervalo en el cual se considera que el fitness sigue siendo efectivamente el mismo (+-`epsilon`).
            - Decimal positivo.
          - `number_of_generations: <int>`
            - Cantidad de generaciones por el cual el mejor fitness de la generacion se debe mantener dentro del `epsilon` para cortar ejecucion.
            - Entero positivo menor igual a 1000.
          - `limit_generation: <int>`
            - Propiedad equivalente al parametro del mismo nombre del metodo de corte **Por Valor de Fitness**
    - **Por Convergencia de la Diversidad**
      - Corta la ejecucion cuando el valor medio de la diversidad de cada atributo de la poblacion se mantiene debajo de un valor determinado durante una cantidad determinada de generaciones.
      - El valor de la diversidad se calcula para cada uno de los 6 atributos que influencian el cálculo del fitness (strength, agility, experience, endurance, vitality, height). Este valor es obtenido calculando el coeficiente de variación respecto a la suma de los atributos de los componentes (no aplicando tanh). Esto permite medir la variación relativa, independiente de la escala. 
        - `name: by_diversity_convergence`
        - `params: `
          - `threshold: <float>`
            - Valor el cual la media de la diversidad debe superar durante una cantidad `number_of_generations` de generaciones para cortar ejecucion.
          - `number_of_generations: <int>`
            - Cantidad de generaciones por el cual la media de la diversidad debe superar el valor de `threshold` para cortar ejecucion.
            - Entero positivo menor igual a 1000.
          - `limit_generation: <int>`
            - Propiedad equivalente al parametro del mismo nombre del metodo de corte **Por Valor de Fitness**.
- `plotting: `
  - Dentro de este mapa se encuentran todas las propiedades encargadas de configurar como se grafican los resultados de cada generacion en tiempo real.
    - `render: <bool>` *Opcional*
      - Si este parametro es `false`, entonces la simulacion no graficara nada.
      - El valor por default es `true`
    - `step: <int>` *Opcional*
      - Este parametro define la cantidad de generaciones que deben ser procesadas antes de re-dibujar el grafico para mostrar la informacion actualizada.
      - Puede servir para evitar que el grafico se re-dibuje constantemente, dificultando su lectura. Ademas, mejora la performance, re-dibujando solo cuando hay suficiente informacion nueva que mostrar para que valga la pena el costo del re-dibujado.
      - Por default, este valor es 10
    - `process_gen_interval: <int>`
      - Delay (en milisegundos) entre ejecucion de procesado de nueva generacion. Si este delay es muy alto, el grafico se calculara muy lento, sin embargo, de ser muy alto, el grafico puede no tener suficiente tiempo para re-dibujarse entre iteraciones, por lo que se muestra vacio.
      - Especialmente util aumentar este valor con valores de `step` bajos.

Ejemplos de configuracion

```yaml
TODO EJEMPLO 1
```
```yaml
TODO EJEMPLO 2
```

### Resultado

TODO: RESULTADOS
