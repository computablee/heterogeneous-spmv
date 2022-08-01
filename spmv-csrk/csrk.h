/* Contains the matrix(graph) structures, including functions to modify the
 * matrices and put them in proper format for performing SpMV and TriSolve
 * Author: Humayun Kabir ( kabir@psu.edu )*/

#ifndef matrix_util_h
#define matrix_util_h
//#define DEBUG

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>

#include <algorithm>
#include <ctime>
#include <sys/time.h>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>

//#define USE_HBW

#if __AVX512F__ == 1
#pragma message "*** AVX512 enabled ***"
#endif

#ifdef USE_HBW
#include <hbwmalloc.h>
#else
#define hbw_malloc malloc
#define hbw_free free
#endif

//#define DEBUG
using namespace std;

const int SUPER_ROW_SIZE = 80;
typedef float val_type;
typedef unsigned int eid_t;
typedef unsigned int vid_t;

typedef std::vector<unsigned int> vid_vec;

// Parallel Ordering types
const string COLOR = "COLOR";
const string LS = "LS";

// Sparse kernel types
const string SpMVKernel = "SpMV";
const string STSKernel = "STS";

// MArker vertex used in level-set
const int VERTEX_MARKER = -10;

// Different coarsening type
const string HAND = "HAND";
const string HEM = "HEM";
const string LEM = "LEM";
const string RAND = "RAND";

struct adj_deg {
  int adj;
  int deg;

  adj_deg() {
    adj = 0;
    deg = 0;
  }

  adj_deg(int a, int d) {
    adj = a;
    deg = d;
  }
};

/*
bool compAdjDeg(adj_deg adjL, adj_deg adjR)
{
  return adjL.adj <= adjR.adj;
}
*/
struct degree_id_pair {
  int deg;
  int id;

  degree_id_pair() {
    deg = 0;
    id = 0;
  }

  degree_id_pair(int a, int b) {
    deg = a;
    id = b;
  }
};

/*
bool compare_deg_id_pair(degree_id_pair p1, degree_id_pair p2)
{
  return p1.deg < p2.deg;
} */

// Renumber the vertices according to a reordering
void renumberGraphUsingReorderedVertices(long numVerts, long NNZ,
                                         unsigned int **inrVec,
                                         unsigned int **incVec,
                                         unsigned int **inDegree,
                                         unsigned int *oldToNewPerm,
                                         unsigned int *newToOldPerm);

/* Graph Coloring */
void BGL_ordering(int nnz, int nRows, int *num_edges, int *adj, size_t *p,
                  int *numColors, int **colorPtr);

// Find levels given a graph
void find_levels(int nnz, int nRows, int *num_edges, int *adj,
                 int *&adjLevelSet, int &numLevel, int *&numEdgesLevelSet);
void find_levels_from_maxDegree_vertex(int nnz, int nRows, int *num_edges,
                                       int *adj, int *&adjLevelSet,
                                       int &numLevel, int *&numEdgesLevelSet);

/* Functions related to RCM ordering */
int rcm_reordering(int num_verts, int *num_edges, int *adj_list, int root,
                   int mask[], int oldToNewOrdering[], int newToOldOrdering[],
                   int &firstVtxinBFS, int &lastVtxinBFS, int &ccSize);

int rcm_reordering_g(int num_verts, int *num_edges, int *adj_list,
                     int *adj_degree, int root, int mask[],
                     int oldToNewOrdering[], int newToOldOrdering[],
                     int &firstVtxinBFS, int &lastVtxinBFS, int &ccSize);

void findPseudoPeripheralVertex(int &root, int *r_vec, int *c_vec, int *mask,
                                int &num_lvls, int *r_vec_lvlStruc,
                                int *c_vec_lvlStruc);
void findRootedLevelStructures(int root, int *r_vec, int *c_vec, int *mask,
                               int &num_lvls, int *r_vec_lvlStruc,
                               int *c_vec_lvlStruc);

// int RCMonSupNodes(graph_t *g, graph_bigAndMapping *g_big);

// Read config file
void readConfigFile(char *configFileName, string &kernelType,
                    string &orderingType, string &corseningType, int &k,
                    int *&superRowSizes);

// Structure to represent an edge
struct Edge {
  unsigned int u;
  unsigned int v;

  Edge(unsigned int inU, unsigned int inV) : u(inU), v(inV) {}

  unsigned int either() const { return u; }
  unsigned int other(unsigned int inU) const {
    if (inU == u)
      return v;
    else
      return u;
  }

  bool operator<(const Edge right) const {
    if (u < right.u)
      return true;
    else if (v < right.v)
      return true;
    return false;
  }
};

template <typename T1, typename T2> struct sortBySecond {
  typedef pair<T1, T2> typePair;
  bool operator()(typePair const &a, typePair const &b) const {
    return a.second < b.second;
  }
};

class BAND_k;

class C_GRAPH {
  friend class BAND_k;

  static int numCopy;

private:
  long N;
  long NNZ;

  unsigned int *r_vec;
  unsigned int *c_vec;
  unsigned int *degree;
  unsigned int *edgeWeight;
  unsigned int *vertexWeight;

public:
  C_GRAPH() {
    numCopy++;
    N = 0;
    NNZ = 0;

    r_vec = NULL;
    c_vec = NULL;
    degree = NULL;
    vertexWeight = NULL;
    edgeWeight = NULL;
  }

  C_GRAPH(long inN, long inNNZ, unsigned int *rVec, unsigned int *cVec,
          unsigned int *inDegree) {
    numCopy++;

    N = inN;
    NNZ = inNNZ;

    r_vec = rVec;
    c_vec = cVec;
    degree = inDegree;
  }

  ~C_GRAPH() {
#ifdef DEBUG
    cout << "Destructor of C_GRAPH called\n";
#endif
    if (numCopy > 0) {

      if (r_vec != NULL)
        delete[] r_vec;

      if (c_vec != NULL)
        delete[] c_vec;

      if (degree != NULL)
        delete[] degree;

      if (edgeWeight != NULL)
        delete[] edgeWeight;

      if (vertexWeight != NULL)
        delete[] vertexWeight;

      numCopy--;
    }

#ifdef DEBUG
    cout << "Destructor of C_GRAPH Completed\n";
#endif
  }
};

// int C_GRAPH::numCopy =0;

class CSRk_Graph {
  friend class BAND_k;

private:
  // The input matrix data
  long N;
  long M;
  long NNZ;

  unsigned int *r_vec;
  unsigned int *c_vec;
  float *val;

  // The value of k
  int k;
  unsigned int *supRowSizes;

  // Number of coarsest rows
  long numCoarsestRows;

  // Need k mappings from corarsest rows to finest rows
  unsigned int **mapCoarseToFinerRows;
  unsigned int **mapCoarseToFinerRows_gpu;

  // k-1 permutations of finer graphs vertices corresponsing to mappings
  unsigned int **permOfFinerRows;

  // Permutation of the original matrix
  unsigned int *permBigG;

  float *x_test;

  // Flags to indicate Kernel and if preprocessed or not
  string kernelType;
  bool ifTuned;

  // Coarsening type
  string coarsenType;

  // For STS only
  string orderType;
  int numPacks;
  int *packsPointer;

  unsigned int *num_edges_L;
  unsigned int *adj_L;
  float *val_L;

  unsigned int *num_edges_U;
  unsigned int *adj_U;
  float *val_U;

  float *x;
  float *b;

public:
  CSRk_Graph();
  CSRk_Graph(long nRows, long nCols, long nnz, unsigned int *rVec,
             unsigned int *cVec, float *value, string kernelCalled,
             string inOrderType, string inCoarsenType, bool isTuned, int inK,
             int *inSupRowSizes);

  // Define copy constructor here
  // CSRk_Graph(const CSRk_Graph& rhs);

  ~CSRk_Graph();

  void putInCSRkFormat();

  // Renumbers the matrix according to the new parmutation
  void reorderA();

  unsigned int *getPermutation() { return permBigG; }

  void setX(float *x) {
    if (k == 1) {
      for (int i = 0; i < N; i++)
        x_test[i] = x[i];
    } else {
      for (int i = 0; i < N; i++)
        x_test[i] = x[permBigG[i]];
    }
  }

  // void SpMV(float *x, float **y);
  void SpMV(float *&y);

  // For STS
  void lowerSTS();
  void incomplete_choloskey();
  void compute_b();
  void checkError();
};

class BAND_k {
private:
  int k;
  C_GRAPH *smallGraphs;

public:
  BAND_k() {
    k = 0;
    smallGraphs = NULL;
  }

  BAND_k(int inK) {
    k = inK;
    smallGraphs = new C_GRAPH[k];
  }

  ~BAND_k() {
#ifdef DEBUG
    cout << "Destructor of BAND_k called\n";
#endif

    if (k > 0) {
      delete[] smallGraphs;
    }
#ifdef DEBUG
    cout << "Destructor of BAND_k Completed\n";
#endif
  }

  void preprocessingForSpMV(CSRk_Graph &csrkGraph);
  void preprocessingForSTS(CSRk_Graph &csrkGraph);

  void stsPreprocessingWithMatching(CSRk_Graph &csrkGraph);
  void stsPreprocessingForHAND(CSRk_Graph &csrkGraph);

  // Coarsen the graph using hand coarsening/HEM
  void coarsenTheGraph(int level, int supRowSizeInRows, int supRowSizeInNNZ,
                       CSRk_Graph &csrkGraph);

  // Hand coarsen the graph such that each super row has super_node_nnz
  // non-zeros
  void handCoarsen(int level, int super_node_nnz, CSRk_Graph &csrkGraph);

  // Coarsen the graph using different machings such that each super row has
  // superNodeSize rows
  void coarsenUsingMatching(int level, int superNodeSize,
                            CSRk_Graph &csrkGraph);

  // Coarsen a graph using random matching
  void randomMatching(C_GRAPH g, unsigned int *perm, C_GRAPH &coarsenedG);

  // Coarsen a using Heavy Edge Matching
  void heavyEdgeMatching(C_GRAPH g, unsigned int *perm, C_GRAPH &coarsenedG);

  // Coarsen a graph using Light Edge Matching
  void lightEdgeMatching(C_GRAPH g, unsigned int *perm, C_GRAPH &coarsenedG);

  // Uncoarsen the coarsened graph and find permutation of original matrix/graph
  // void reorder_matrix_A( unsigned int *smallG_permutation );

  void uncoarsenTheGraph(int level, CSRk_Graph &csrkGraph,
                         unsigned int *permCorserGraph,
                         unsigned int *&permFinerGraph);

  void matchingUncoarsenTheGraph(int level, CSRk_Graph &csrkGraph);
};

struct Pair {
  unsigned int first;
  unsigned int second;

  Pair() {
    first = 0;
    second = 0;
  }

  Pair(unsigned int a, unsigned int b) {
    first = a;
    second = b;
  }
};

/*
bool myComparePair(Pair p1, Pair p2)
{
  return p1.first < p2.first;
}
*/
struct rev_deg_id_pair {
  unsigned deg;
  unsigned id;

  rev_deg_id_pair() {
    deg = 0;
    id = 0;
  }

  rev_deg_id_pair(unsigned a, unsigned b) {
    deg = a;
    id = b;
  }
};

/*
bool compare_rev_deg_id_pair(rev_deg_id_pair p1, rev_deg_id_pair p2)
{
  return p1.deg > p2.deg;
}
*/

struct Pair_CSR {
  unsigned first;
  unsigned second;

  Pair_CSR() {
    first = 0;
    second = 0;
  }

  Pair_CSR(unsigned a, unsigned b) {
    first = a;
    second = b;
  }
};

/*
bool myCompare(Pair_CSR p1, Pair_CSR p2)
{
  return p1.first < p2.first;
}

*/

struct supNodeToIndexMap {
  unsigned int supNodeIndex;
  int indexInSuperNode;
};

class Queue_Arry {

  int size;
  int start;
  int end;
  int *vertexQueue;

public:
  Queue_Arry() {
    start = 0;
    end = 0;
    size = 0;
  }

  Queue_Arry(int in_size) {
    start = 0;
    end = 0;
    size = in_size;
    vertexQueue = new int[in_size];
  }

  ~Queue_Arry() { delete[] vertexQueue; }

  void resize(int in_size) {

    start = 0;
    end = 0;
    if (size > 0)
      delete[] vertexQueue;
    size = in_size;
    vertexQueue = new int[in_size];
  }

  void enQueue(int inVtx) {

    if (!overFlow()) {
      vertexQueue[end] = inVtx;
      end++;
    } else {
      printf("Size of Queue: %d\n", size);
      printf("Queue overFlow, can't enqueue: Queue\n");
      exit(1);
    }
  }

  int deQueue() {
    if (!isEmpty()) {
      start++;
      return vertexQueue[start - 1];
    } else {
      printf("Queue underFlow, can't dqueue: Queue\n");
      exit(1);
    }
  }

  bool isEmpty() { return start >= end; }

  bool overFlow() { return end >= size; }
};

template <class T> struct Node {
  T val;
  Node *next;
};

template <class T> class Queue {

private:
  Node<T> *head;
  Node<T> *tail;

  void insert(T inVal);
  T Delete();

public:
  Queue();
  ~Queue();
  void Enqueue(T inVal);
  T Dequeue();

  bool isEmpty();
};

template <class T> Queue<T>::~Queue() {

  while (head->next != NULL) {
    Node<T> *CurHead = head;
    delete (CurHead);
    head = head->next;
  }
  delete (head);
}

template <class T> Queue<T>::Queue() { head = tail = NULL; }

template <class T> void Queue<T>::insert(T inVal) {
  Node<T> *newNode = new Node<T>();

  newNode->val = inVal;
  newNode->next = NULL;

  if (tail == NULL) {
    head = tail = newNode;
  } else {
    tail->next = newNode;
    tail = newNode;
  }
}

template <class T> T Queue<T>::Delete() {
  try {
    if (head == NULL) {
      throw string("Trying to delete from Empty Queue .");
    } else {
      Node<T> *curHead = head;
      T retVal = head->val;
      head = head->next;

      if (head == NULL)
        tail = NULL;

      delete (curHead);
      return retVal;
    }
  } catch (string s) {
    cout << "Exception: " << s << endl;
  } catch (...) {
    cout << " An exception occurred. \n";
  }
}

template <class T> void Queue<T>::Enqueue(T inVal) { insert(inVal); }

template <class T> T Queue<T>::Dequeue() { return Delete(); }

template <class T> bool Queue<T>::isEmpty() { return head == NULL; }

// find a mapping from coarsest graph vertices to finest graph vertices
void findFinalMapping(int numSupRows, vector<unsigned int> &vectorSizes,
                      vector<unsigned int *> &mappingVectors,
                      unsigned int *&numEdgesSupRowsToRows,
                      unsigned int *&mapSupRowstoRows);

#endif
