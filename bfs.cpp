#include <igraph.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>

using namespace std;
using std::cout;

#define ROOT 0

vector<int> split(string& s, const string& delimiter) {
  vector<int> tokens;
  size_t pos = 0;
  string token;
  while ((pos = s.find(delimiter)) != string::npos) {
    token = s.substr(0,pos);
    tokens.push_back(atoi(token.c_str()));
    s.erase(0, pos + delimiter.length());
  }
  tokens.push_back(atoi(s.c_str()));
  return tokens;
}

int main(int argc, char* argv[]) {
  int my_rank , nprocs;
  int my_work;
  int no_of_nodes = 34546;
  int start = 9907233;
  int end = 9301253;
  char buffer[100];
  int num_vertices = 0;
  //map<int, int> id_map;
  
  /*input = fopen("./dataset.tsv", "r");
  ofstream outputFile("./dataset_mapped.tsv");
  ifstream inputFile("./dataset.tsv");
  string line;
  string delimiter = "\t";
  vector<int> edges;
  int current = 0;
  int from, to;
  while(getline(inputFile,line)) {
    edges = split(line, delimiter);
    from = edges[0];
    to = edges[1];
    if (auto search = id_map.find(edges[0]); search == id_map.end()) {
      id_map[from] = current;
      current += 1;
    }
    if (auto search = id_map.find(edges[1]); search == id_map.end()) {
      id_map[to] = current;
      current += 1;
    }
    if (outputFile.is_open()) {
      outputFile << id_map[from];
      outputFile << delimiter;
      outputFile << id_map[to];
      outputFile << "\n";
    }
  }

  printf("nodes mapped");

  if (current != no_of_nodes) {
    printf("no_of_nodes %d does not match current %d\n", no_of_nodes, current);
  }
  */
  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  igraph_bool_t directed = true;

  if (my_rank == ROOT) {
    //mapped = fopen("./dataset_mapped.tsv", "r");
    FILE* input = fopen("./dataset.tsv", "r");
    printf("File opens\n");
    igraph_t* graph;
    igraph_read_graph_edgelist(graph, input, 0, directed);
    num_vertices = igraph_vcount(graph);
    printf("num of vertices: %d\n",num_vertices);
  }
  MPI_Finalize();
  return 0;
}
