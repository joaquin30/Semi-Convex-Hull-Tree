#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "schtree.h"

using namespace std;

constexpr int DIM = 13;
using ScalarT = float;
using vec = sch::Vec<DIM, ScalarT>;
using node = sch::Node<DIM, ScalarT>;
using knn_result = sch::KnnResult<ScalarT>;

knn_result knnSearchBruteforce(const vector<vec>& vecs, const vec& q, const int k)
{
    int n = vecs.size();
    knn_result res(k);
    for (int i = 0; i < n; ++i)
    res.insert(i, sch::distance(q, vecs[i]));

    res.sort();
    return res;
}

void readDataset(const char* filename, vector<string>& name, vector<vec>& data)
{
    ifstream file(filename);
    if (file.fail()) {
        cerr << "ERROR: Archivo \"" << filename <<  "\" no encontrado\n";
        exit(1);
    }

    string val;
    vec tmp;
    getline(file, val);
    while (file.peek() != EOF) {
        for (int i = 0; i < 13; ++i) {
            getline(file, val, ',');
            tmp[i] = stof(val);
        }
        
        data.push_back(tmp);
        getline(file, val);
        val.pop_back();
        name.push_back(val);
    }
}

int main(int argc, char* argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.precision(4);
    cout << fixed;
    srand(time(nullptr));

    if (argc != 2) {
        cerr << "ERROR: Uso del comando:\n"
             << "\ttest.exe <ruta_al_dataset>\n";
        exit(1);
    }
    
    vector<string> name;
    vector<vec> data;
    readDataset(argv[1], name, data);
    sch::Tree<DIM, ScalarT> tree(data);
    tree.assertLeafs();
    int k = 100; // si k es mayor se demora mucho
    for (const auto& pnt : data) {
        auto res1 = tree.knnSearch(pnt, k, true);
        auto res2 = knnSearchBruteforce(data, pnt, k);
        assert(res1.size() == k);
        assert(res2.size() == k);
        assert(res1 == res2);
    }

    for (int i = 0; i < 1000; ++i) {
        vec pnt;
        for (int j = 0; j < DIM; ++j)
            pnt[j] = (float) rand() / RAND_MAX * 100.f;

        auto res1 = tree.knnSearch(pnt, k, true);
        auto res2 = knnSearchBruteforce(data, pnt, k);
        assert(res1.size() == k);
        assert(res2.size() == k);
        assert(res1 == res2);
    }

    cout << "OK\n";
    return 0;
}
