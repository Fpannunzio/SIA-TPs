# Sokoban Solver

Sokoban Solver es un programa implementado en Python3 para resolver niveles del juego [Sokoban](https://en.wikipedia.org/wiki/Sokoban) utilizando los siguientes algoritmos de busqueda.

- Desinformados
  - BFS - Breadth First Search
  - DFS - Depth First Search
  - IDDFS - Iterative Deepening Depth First Search
- Informados
  - Greedy
  - A* - A Star
  - IDA - Iterative Deepening A Star

## Autores

- [Brandy, Tobias](https://github.com/tobiasbrandy)
- [Pannunzio, Faustino](https://github.com/Fpannunzio)
- [Sagues, Ignacio](https://github.com/isagues)

## Dependencias

- Python 3.6+
- PyGame 2+
- PyYAML 5+

Contando con python3 instalado se pueden instalar las dependencias restantes con `pip install pygame pyyaml`

## Ejecucion

Para ejecutar el programa no es necesario hacer ningun tipo de compilacion, basta con ejecutarlo.

Para esto se debera correr `python sokoban_solver.py`. 

A menos que se indique lo contrario, se buscara en la raiz del proyecto el archivo `config.yaml` donde debera estar toda la informacion de [configuracion](#configuracion). 

En caso de querer almacenar el archivo de configuracion en otro ubicacion o con otro nombre se debera proveer el path del mismo como argumento de ejecucion `python sokoban_solver.py <path>`.

Una vez finalizado se abrira de forma automatica una ventana mostrando el juego y la secuencias de pasos correspondiente a la solucion.

### Configuracion

La configuracion de ejecucion es realizada via un archivo de tipo [YAML](https://yaml.org/) el cual debera contener la siguiente informacion.

- `level: <path_to_level>`
  - **Requerido**.
  - El path es relativo a `assets/levels`. 
  - El archivo tiene que estar en ["notacion estandar"](https://docs.ansible.com/ansible/2.3/YAMLSyntax.html) de Sokoban.
- `strategy: `
  - `name: <strategy_name>`
    - **Requerido**
    - Contiene el nombre del algoritmo a utilizar para resolver el nivel.
    - Los valores posibles son `BFS`, `DFS`, `IDDFS`, `GREEDY`, `A*`, `IDA`.
  - `params:`
    - `step: int`
      - Solo es utilizado con `IDDFS`.
      - Espera un valor numerico.
      - El valor por defecto es `10`.
    - `filter_lost_states: bool`
      - Aplica solo a los algoritmos desinformados (`BFS`, `DFS`, `IDDFS`).
      - Permite filtrar aquellos estados en los cuales es imposible ganar ya que una caja fue movida a una esquina de paredes.
      - El valor por defecto es `True`.
    - `heuristic: <heuristic_name>`
      - Aplica solo a los algoritmos informados (`GREEDY`, `A*`, `IDA`). **Requerido** en estos casos.
      - Indica que heuristica sera utilizada. Valores posibles:
        - `target_box_dist`: Manhattan Distance minima entre las cajas y los targets.
        - `player_box_dist`: Manhattan Distance entre el jugador y su caja mas cercana.
        - `open_goal`: Cantidad de targets por completar.
        - `target_box_dist_plus_open_goal`: Suma entre `target_box_dist` y `open_goal`.
- `render: bool`
  - Indica si se debera mostrar la solucion una vez que sea alcanzada.
  - Por defecto es `True`

Ejemplo de configuracion

```yaml
level: level1.txt
strategy:
  name: IDDFS
  params:
    step: 25
    filter_lost_states: true
```

### Resultado

El resultado de la ejecucion es impreso por salida estandar una vez finalizada la ejecucion. Esta contiene informacion que podria ser de utilidad a la hora de analizar el comportamiento de los distintos algoritmos de busqueda.

- `Total runtime`: Tiempo que se tardo en correr el algoritmo.
- `Total moves`: Cantidad de pasos que contiene la solucion encontrada. Teniendo en cuesta que es un problema de costo uniforme, tambien indica el costo de la solucion.
- `Total exploded nodes`: Cantidad de nodos que fueron analizados y expandidos para poder analizar a sus hijos.
- `Total leaf nodes`: Cantidad de nodos hoja al finalizar la ejecucion.

Ademas, se informa si el nivel seleccionado no posee solucion.

## Reconocimientos

Tanto la inspiracion para la logica de rendering de la solucion, como los assets visuales utilizados provienen de la implementacion de [Sokoban de Gemkodor](https://github.com/Gemkodor/sokoban)
