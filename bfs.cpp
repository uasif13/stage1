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

vector<int> split(string &s, const string &delimiter)
{
  vector<int> tokens;
  size_t pos = 0;
  string token;
  while ((pos = s.find(delimiter)) != string::npos)
  {
    token = s.substr(0, pos);
    tokens.push_back(atoi(token.c_str()));
    s.erase(0, pos + delimiter.length());
  }
  tokens.push_back(atoi(s.c_str()));
  return tokens;
}

int main(int argc, char *argv[])
{
  int my_rank, nprocs;
  int my_work;
  int no_of_nodes = 34546;
  int start = 9907233;
  int end = 9301253;
  char buffer[100];
  int num_vertices = 0;

  uint64_t *src;
  uint64_t *dst;
  uint64_t total_src;
  uint64_t total_dst;
  // map<int, int> id_map;

  // ofstream outputFile("./dataset_mapped.tsv");
  // ifstream inputFile("./dataset.tsv");
  // string line;
  // string delimiter = "\t";
  // vector<int> edges;
  // int current = 0;
  // int from, to;
  // while (getline(inputFile, line))
  // {
  //   edges = split(line, delimiter);
  //   from = edges[0];
  //   to = edges[1];
  //   if (auto search = id_map.find(edges[0]); search == id_map.end())
  //   {
  //     id_map[from] = current;
  //     current += 1;
  //   }
  //   if (auto search = id_map.find(edges[1]); search == id_map.end())
  //   {
  //     id_map[to] = current;
  //     current += 1;
  //   }
  //   if (outputFile.is_open())
  //   {
  //     outputFile << id_map[from];
  //     outputFile << delimiter;
  //     outputFile << id_map[to];
  //     outputFile << "\n";
  //   }
  // }
  // outputFile.seekp(0, ios::beg);

  // printf("nodes mapped\n");

  // if (current != no_of_nodes)
  // {
  //   printf("no_of_nodes %d does not match current %d\n", no_of_nodes, current);
  // }

  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  igraph_bool_t directed = true;

  // input = fopen("./dataset.tsv", "r");
  //   printf("File opens\n");

  
  if (my_rank == ROOT)
  {
    igraph_t *graph;
    FILE *mapped = fopen("./dataset_mapped.tsv", "r");
    igraph_read_graph_edgelist(graph, mapped, 0, directed);
    total_src = igraph_vcount(graph) + 1;
    total_dst = igraph_ecount(graph) * 2;
    printf("total_src: %d total_dst: %d\n", total_src, total_dst);

    src = new uint64_t[total_src];
    dst = new uint64_t[total_dst];
    igraph_uint_t dst_index = 0;
    int src_index = 0;
    igraph_adjlist_t al;

    igraph_adjlist_init(graph, &al, IGRAPH_ALL, IGRAPH_NO_LOOPS, IGRAPH_NO_MULTIPLE);

    for (igraph_integer_t i = 0; i < total_src - 1; i++)
    {
      igraph_vector_int_t *v_list = igraph_adjlist_get(&al, i);

      src[i] = dst_index;
      printf("src_index: %d dst_index: %d\n",i, dst_index);
      igraph_uint_t al_size = igraph_vector_int_size(v_list);
      // printf("first elem in al: %d\n", VECTOR(*v_list)[0]);
      printf("igraph al size : %d\n", al_size);
      for (igraph_integer_t j = 0; j < al_size; j++)
      {
          // dst[dst_index] = VECTOR(*v_list)[j];
        //   dst_index++;
      }
      dst_index += al_size;
      
    }
    // src[src_index] = dst_index;
  }

  MPI_Finalize();
  return 0;
}
