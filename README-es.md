# Semi-Convex Hull Tree

Una estructura de datos optimizada para la consulta exacta de los K vecinos más cercanos (KNN) de un conjunto de puntos.
Implementación del paper [Semi-Convex Hull Tree: Fast Nearest Neighbor Queries for Large Scale Data on GPUs](https://ieeexplore.ieee.org/document/8594919).
[Repositorio de la implementación oficial](https://github.com/XFastDataLab/Semi-convex_Hull_Tree).

## Dependencias

- [Eigen3](https://eigen.tuxfamily.org)
- OpenMP (opcional)

## Instalación

Simplemente copia `schtree.h` a tu proyecto. Debe estar disponible la
libreria Eigen3. Para instalarla puedes seguir [esta guia](https://eigen.tuxfamily.org/dox/GettingStarted.html).
Opcionalmente puedes activar OpenMP usando tu compilador.

## Ejemplo

~~~cpp
#include <iostream>
#include <vector>
#include <schtree.h>

const int D = 5; // número de dimensiones
const int N = 1000; // número de puntos

int main() {
    std::vector<sch::Vec<D>> points(N); // conjunto de puntos
    sch::Tree<D> tree(points); // la estructura de datos
    sch::Vec<D> query; // el punto a buscar sus k vecinos cercanos
    auto res = tree.knnSearch(query, 30); // buscamos sus 30 vecinos más cercanos
    for (const auto& i : res) {
        std::cout << points[i.idx] << "\n"; // imprimimos los resultados
    }
}
~~~

## Pruebas

Para correr las pruebas usa `make test`. Esto compilará el archivo `test/test.cpp`
y lo ejecutará usando el dataset ubicado en `test/dataset.csv`, que corresponde
a [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify).

**Nota:** Para correr los tests probablemente deberás modificar el `Makefile`.

## Documentación

### namespace sch

Donde están todas las funciones y clases del árbol

### sch::Vec<DIM, Scalar = float>

Es un alias de una matriz de Eigen3 de DIM filas y 1 columna. Por defecto `Scalar` es `float`.

### sch::KnnResult<Scalar = float>

Es el resultado de la búsqueda KNN. Para acceder a los puntos individuales usa:

~~~cpp
for (const auto& i : res) {
    i.idx; // el indice del punto
    i.dist; // la distancia del punto hacia q
}
~~~

### sch::Tree<DIM, Scalar = float>

La estructura de datos. Alias en la clase:

-   `vec`: `Vec<DIM, Scalar>`.
-   `knn_result`: `KnnResult<Scalar>`.

### sch::Tree(std::vector<vec>& points, bool copy = false)

-   `points`: Los puntos para construir el árbol.
-   `copy`: Si es verdadero copia los puntos en el arbol, si es falso solo usa un puntero a los puntos.

### knn_result knnSearch(const vec& q, int k, bool sort = false)

-   `q`: el punto a consultar por sus vecinos más cercanos.
-   `k`: el número de vecinos que queremos retornar.
-   `sort`: Si es verdadero ordena los puntos en el resultado por las distancias ascendentemente, 
    si es falso deja los puntos desordenados.

### void knnBulkSearch(std::vector<knn_result>& query_results, const std::vector<vec>& query_points, int k, bool sort = false)

-   `query_results`: Donde se almacenan los resultados de las consultas.
-   `query_points`: Los puntos a consultar por sus vecinos más cercanos.
-   `k`: El número de vecinos que queremos retornar por cada consulta.
-   `sort`: Si es verdadero ordena los puntos en los resultados por las distancias ascendentemente, 
    si es falso deja los puntos desordenados.
-   **Nota:** `knnSearch` es independiente y thread-safe, por lo que `knnBulkSearch`
    está paralelizado con OpenMP. Consume mucha memoria ya que guarda
    todos los resultados.
