#ifndef SCHTREE_H
#define SCHTREE_H

#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>

#include "Eigen/Dense"

namespace sch {

//~ MATH

template <int DIM, class Scalar = float>
using Vec = Eigen::Matrix<Scalar, DIM, 1>;

template <int DIM, class Scalar = float>
struct Hyperplane {
    Vec<DIM, Scalar> a;
    Scalar b;
};

template <int DIM, class Scalar = float>
struct Constraint {
    Hyperplane<DIM, Scalar> hp;
    bool less_equal;
};

template <int DIM, class Scalar>
Scalar distance(const Vec<DIM, Scalar>& v1, const Vec<DIM, Scalar>& v2) { return (v1 - v2).norm(); }

template <int DIM, class Scalar>
Scalar distance(const Vec<DIM, Scalar>& v, const Hyperplane<DIM, Scalar>& hp) { return abs(v.dot(hp.a) - hp.b); }

template <int DIM, class Scalar>
bool inside(const Vec<DIM, Scalar>& v, const Constraint<DIM, Scalar>& ct)
{
    Scalar d = ct.hp.a.dot(v);
    if (ct.less_equal)
        return d <= ct.hp.b;

    return d >= ct.hp.b;
}


//~ KNN

template <class Scalar = float>
struct Knn {
    int idx;
    Scalar dist;
    
    Knn(int i, Scalar d) : idx(i), dist(d) {}
    
    bool operator<(const Knn<Scalar>& other) const
    {
        if (dist == other.dist)
            return idx < other.idx;
            
        return dist < other.dist;
    }
    
    bool operator==(const Knn<Scalar>& other) const { return dist == other.dist && idx == other.idx; }
};

template <class Scalar = float>
class KnnResult {
    using knn = Knn<Scalar>;
    using iterator = typename std::vector<knn>::iterator;
    using const_iterator = typename std::vector<knn>::const_iterator;
    
    int K;
    std::vector<knn> heap;

public:
    KnnResult() : K(0) {}
    
    KnnResult(int k) : K(0) { setK(k); }

    void setK(int k)
    {
        K = std::max(k, K);
        heap.reserve(K);
    }

    //~ Usamos un heap y un array de bools para que la inserción
    //~ de nuevos elementos y la eliminación del mayor sea O(log(n))
    bool insert(int idx, Scalar dist)
    {      
        if (!full()) {
            heap.emplace_back(idx, dist);
            std::push_heap(begin(), end());
            return true;
        }
        
        if (maxDist() > dist) {
            std::pop_heap(begin(), end());
            heap.back() = knn(idx, dist);
            std::push_heap(begin(), end());
            return true;
        }

        return false;
    }
    
    Scalar maxDist() const { return heap.empty() ? std::numeric_limits<Scalar>::max() : heap[0].dist; }
    
    int size() const { return heap.size(); }
    
    iterator begin() { return heap.begin(); }
    const_iterator begin() const { return heap.cbegin(); }
    
    iterator end() { return heap.end(); }
    const_iterator end() const { return heap.cend(); }

    bool operator==(const KnnResult<Scalar>& other) const { return heap == other.heap; }
    
    void sort() { std::sort_heap(heap.begin(), heap.end()); }
    
    bool full() const { return size() >= K; }
};


//~ NODE

template <int DIM, class Scalar = float>
struct Node {
    using node = Node<DIM, Scalar>;
    using constraint = Constraint<DIM, Scalar>;
    using hyperplane = Hyperplane<DIM, Scalar>;
    
    std::vector<int> idxs; 
    std::vector<constraint> constraints;
    node* left;
    node* right;
    bool is_leaf;
    
    Node(const std::vector<int>& vi, const std::vector<constraint>& vc = {})
        : idxs(vi), constraints(vc), left(nullptr), right(nullptr), is_leaf(true) {}
};

template <int DIM, class Scalar = float>
Scalar distance(const Vec<DIM, Scalar>& pnt, Node<DIM, Scalar>* node) {
    Scalar dist = 0;
    for (const auto& ct : node->constraints) {
        if (!inside(pnt, ct))
            dist = std::max<Scalar>(dist, distance(pnt, ct.hp));
    }

    return dist;
}


//~ TREE

template <int DIM, class Scalar = float>
class Tree {
    using vec = Vec<DIM, Scalar>;
    using node = Node<DIM, Scalar>;
    using knn_result = KnnResult<Scalar>;
    using constraint = Constraint<DIM, Scalar>;
    using hyperplane = Hyperplane<DIM, Scalar>;
    
    const int MAX_LEAF_PNTS, NUM_PNTS;
    node* root;
    std::vector<node*> leafs;
    std::vector<vec> data;
    vec* point;

    void deleteTree(node* tree)
    {
        if (!tree)
            return;
        
        deleteTree(tree->left);
        deleteTree(tree->right);
        delete tree;
    }

    //~ Acerca los hyperplanos lo más cerca posible a los puntos, formando
    //~ un "semi convex hull"
    void refineConstraints(node* tree)
    {        
        for (auto& ct : tree->constraints) {
            Scalar dist = std::numeric_limits<Scalar>::max();
            int idx = -1;
            for (int i : tree->idxs) {
                if (!inside(point[i], ct))
                    ct.hp.b = ct.hp.a.dot(point[i]);
            }
            
            for (int i : tree->idxs) {
                Scalar d = distance(point[i], ct.hp);
                if (d < dist) {
                    dist = d;
                    idx = i;
                }
            }
            ct.hp.b = ct.hp.a.dot(point[idx]);
        }
    }

    //~ Divide recursivamente cada nodo en dos hijos
    void splitTree(node* tree)
    {
        if ((int) tree->idxs.size() <= MAX_LEAF_PNTS) {
            refineConstraints(tree);
            return;
        }

        tree->is_leaf = false;
        //~ Acá el paper dice que debemos elegir un punto random, pero elegir
        //~ el primero ahorra memoria y funciona casi igual de bien (creo)
        int ix = tree->idxs[0], ip = -1;
        Scalar dist = -1;
        for (int i : tree->idxs) {
            if (i == ix) continue;
            Scalar d = distance(point[i], point[ix]);
            if (dist < d) {
                ip = i;
                dist = d;
            }
        }

        int iq = -1;
        dist = -1;
        for (int i : tree->idxs) {
            if (i == ip) continue;
            Scalar d = distance(point[i], point[ip]);
            if (dist < d) {
                iq = i;
                dist = d;
            }
        }

        vec a = (point[ip] - point[iq]).normalized();
        Scalar b = a.dot((point[ip] + point[iq]) / 2.f);
        hyperplane hp{a, b};
        std::vector<int> left, right;
        for (int i : tree->idxs) {
            if (a.dot(point[i]) <= b)
                left.push_back(i);
            else
                right.push_back(i);
        }
        
        //~ en caso todos los puntos sean iguales
        if (left.empty() || right.empty()) {
            tree->is_leaf = true;
            refineConstraints(tree);
            return;
        }
        
        tree->left = new node(left, tree->constraints);
        tree->right = new node(right, tree->constraints);
        tree->left->constraints.push_back(constraint{hp, true});
        tree->right->constraints.push_back(constraint{hp, false});
        
        //~ Para liberar memoria
        tree->constraints.clear();
        tree->constraints.shrink_to_fit();
        tree->idxs.clear();
        tree->idxs.shrink_to_fit();
        
        splitTree(tree->left);
        splitTree(tree->right);
    }
    
    //~ Recursivamente obtenemos las hojas del arbol
    std::vector<node*> genLeafs(node* tree)
    {
        if (tree->is_leaf)
            return std::vector<node*>{tree};

        std::vector<node*> ret;
        auto v1 = genLeafs(tree->left);
        auto v2 = genLeafs(tree->right);
        for (auto ptr : v1)
            ret.push_back(ptr);

        for (auto ptr : v2)
            ret.push_back(ptr);
        
        return ret;
    }
    
public:
    Tree(std::vector<vec>& points = {}, bool copy = false)
        : MAX_LEAF_PNTS(std::max<int>(points.size() * 0.01, 10)), NUM_PNTS(points.size())
    {
        if (copy) {
            data = points;
            point = data.data();
        } else {
            point = points.data();
        }

        std::vector<int> idxs(points.size());
        for (int i = 0, n = points.size(); i < n; ++i)
            idxs[i] = i;
        
        root = new node(idxs);
        splitTree(root);
        leafs = genLeafs(root);
    }

    ~Tree() { deleteTree(root); }

    //~ std::vector<node*> getLeafs() { return leafs; }

    //~ Un poco diferente al algoritmo del paper, ordenamos las hojas según su distancia hacia el punto
    //~ usando los constraints y empezamos a iterar, si es que el nodo es mas lejano que la maxima distancia
    //~ de knn result, salimos del bucle
    knn_result knnSearch(const vec& q, int k, bool sort = false)
    {
        knn_result res(k);
        int n = leafs.size();
        std::vector<Knn<Scalar>> query_leafs;
        for (int l = 0; l < n; ++l)
            query_leafs.emplace_back(l, distance(q, leafs[l]));
        
        std::sort(query_leafs.begin(), query_leafs.end());
        for (const auto& leaf : query_leafs) {
            if (!res.full() || leaf.dist < res.maxDist()) {
                for (int i : leafs[leaf.idx]->idxs)
                    res.insert(i, distance(point[i], q));
            } else {
                break;
            }
        }
        
        if (sort)
            res.sort();
        
        return res;
    }

    void knnBulkSearch(std::vector<knn_result>& query_results,
        const std::vector<vec>& query_points, int k, bool sort = false)
    {
        int n = query_points.size();
        query_results.resize(n);
        #pragma omp parallel for shared(query_points, query_results, n)
        for (int i = 0; i < n; ++i)
            query_results[i] = knnSearch(query_points[i], k, sort);
    }

    //~ comprobamos con assertions que todos los puntos se encuentren adentro de las
    //~ constraints de las hojas
    void assertLeafs()
    {
        int sz = 0;
        for (const auto& leaf : leafs) {
            for (const auto& ct : leaf->constraints) {
                for (int i : leaf->idxs)
                    assert(inside(point[i], ct));
            }
            sz += leaf->idxs.size();
        }
        
        assert(sz == NUM_PNTS);
    }
};

} // namespace sch

#endif // SCHTREE_H
