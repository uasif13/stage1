/*
  bfs mpi no cuda path
  compile: mpicxx bfs_mpi_no_cuda_path.cpp -I/usr/local/include/igraph -L/usr/local/lib -ligraph -o bfs_mpi_no_cuda_path
  run: mpirun -n <nprocs> astar <grid_size> <start_node> <end_node> <heuristic>
 */
#include <igraph.h>
#include <mpi.h>
#include <cmath>
#include <math.h>
#include <climits>
#include <vector>
#include <array>
#include <iostream>
#include <sys/time.h>
#include <random>
#include <fstream>
#include <string>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#define MASTER 0
#define ROOT 0
/*
  nVVBuf(local) - 0 for no update, 1 for update
  checkBuf(global) - shared nVV
  srcPtrs, dst, srcPtrs_size, dst_size -> graph
  levelBuf - iteration level for vertex
 */
#define MAX_BUF_SIZE 1<<25
uint64_t nVVBuf[MAX_BUF_SIZE], levelBuf[MAX_BUF_SIZE];
igraph_t * global_graph;
uint64_t checkBuf[MAX_BUF_SIZE];

void output_vec(uint64_t * data, int data_size, int my_rank) {
  printf("my_rank: %d: ", my_rank);
  for (int i = 0; i < data_size; i++)
    printf("%*" PRIu64" ", 10, data[i]);
  printf("\n");
}

void output_graph(uint64_t* data, uint64_t grid_size, int my_rank) {
  printf("my_rank: %d \n", my_rank);
  for (int i = 0; i < grid_size*grid_size; i++) {
    if (i % grid_size == 0)
      printf("\n");
    printf("%*" PRIu64 " ", 5 , data[i]);
  }
  printf("\n");
}

bool checkNVV(uint64_t * checkBuf, int nprocs) {
  for (int i = 0; i < nprocs; i++) {
    if (checkBuf[i] == 1)
      return true;
  }
  return false;
    
}
// for open and closed
void aggregate_open_or_closed(uint64_t * buffer_recv, uint64_t* buffer, int my_rank, int my_work, int nprocs) {
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++) {
    for (uint64_t j = 0; j < nprocs; j++) {
      uint64_t buffer_recv_index = i + j*my_work;
      uint64_t buffer_index = i + my_rank*my_work;
      if (buffer_recv[buffer_recv_index] == 1 && buffer[buffer_index] == 0) {
	buffer[buffer_index] = 1;
	nVVBuf[my_rank] = 1;
      }
    }
  }		  
}
// for g List
void aggregate(uint64_t * buffer_recv, uint64_t* buffer, int my_rank,  int my_work, int nprocs) {
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++) {
    for (uint64_t j = 0; j < nprocs; j++ ) {
      uint64_t buffer_recv_index = i + j*my_work;
      uint64_t buffer_index = i+my_rank*my_work;
      if (buffer_recv[buffer_recv_index] != 0 && buffer_recv[i+j*my_work] <buffer[i+my_rank*my_work]) {
	buffer[i+my_rank*my_work] = buffer_recv[i+j*my_work];
	//	printf("my_rank: %d buffer_index: %*" PRIu64 " %*" PRIu64 "\n", my_rank,10, buffer_index, 10, buffer_recv_index );
	nVVBuf[my_rank] = 1;
	//	printf("my_rank: %d levelBuf: %d bufflevelrecv: %d nVV: %d\n", my_rank, i+my_rank*my_work, j*my_work, nVVBuf[my_rank]);
      }
    }
  }
}
// for parent
void aggregate_parent(uint64_t * buffer_recv, uint64_t* buffer, uint64_t *gList, int my_rank,  int my_work, int nprocs) {
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++) {
    for (uint64_t j = 0; j < nprocs; j++ ) {
      uint64_t buffer_recv_index = i + j*my_work;
      uint64_t buffer_index = i+my_rank*my_work;
      if (buffer_recv[buffer_recv_index] != buffer[buffer_index] && gList[buffer_recv[buffer_recv_index]] < gList[buffer[buffer_index]]) {
	buffer[buffer_index] = buffer_recv[buffer_recv_index];
	//	printf("my_rank: %d buffer_index: %*" PRIu64 " %*" PRIu64 "\n", my_rank,10, buffer_index, 10, buffer_recv_index );
	nVVBuf[my_rank] = 1;
	//	printf("my_rank: %d levelBuf: %d bufflevelrecv: %d nVV: %d\n", my_rank, i+my_rank*my_work, j*my_work, nVVBuf[my_rank]);
      }
    }
  }
}

int main(int argc, char * argv[]) {
  int my_rank, nprocs;
  uint64_t my_work;
  uint64_t no_of_nodes;
  
  uint64_t start, end;

  uint64_t * src;
  uint64_t * dst;
  uint64_t total_src;
  uint64_t total_dst;

  if (argc == 3) {
    start = atoi(argv[1]);
    end = atoi(argv[2]);
  }else {
    start = 9907233;
    end = 9301253;
  }

  struct timeval timecheck;
  
  uint64_t nVV;
    
  MPI_Comm world = MPI_COMM_WORLD;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // Initialization
  no_of_nodes = 34546;
  printf("no_of_nodes: %d\n", no_of_nodes);
  uint64_t * hList = new uint64_t[no_of_nodes];
  uint64_t * gList = new uint64_t[no_of_nodes];
  uint64_t * openList = new uint64_t[no_of_nodes];
  uint64_t * closedList = new uint64_t[no_of_nodes];
  uint64_t * parent = new uint64_t[no_of_nodes];
  uint64_t * dataset_id_to_graph_id = new uint64_t[no_of_nodes];

  igraph_bool_t directed = true;
  
  for (uint64_t i = 0; i < no_of_nodes; i++) {
    hList[i] = 0;
    gList[i] = numeric_limits<uint64_t>::max();
    openList[i] = 0;
    closedList[i] = 0;    
    parent[i] = numeric_limits<uint64_t>::max();
    dataset_id_to_graph_id[i] = 0;
  }
    
  printf("arrays initialized\n");
  // TODO: Add check for valid node
  
  printf("my_rank: %d start: %d, end: %d\n", my_rank, start, end);
  if (my_rank == ROOT) {
    igraph_t*  graph;
    // Create graph
    
    FILE* input;
    input= fopen("./dataset.tsv", "r");
    
        igraph_read_graph_edgelist(graph, input, 0, directed );
	printf("dataset read\n");
	//igraph_vector_int_t *v_list;
	//igraph_vs_t all_v_list = igraph_vs_all(igraph_vss_vector(v_list)));
	//printf("first mode: ", VECTOR(*v_list)[0], "\n ");
	/*
	total_src = igraph_vcount(&graph)+1;
	total_dst = igraph_ecount(&graph)*2;
	printf("my_rank: %d total_src: %*" PRIu64 " total_dst %*" PRIu64 "\n", my_rank, 10, total_src,10, total_dst);
      src = new uint64_t[total_src];
      dst = new uint64_t[total_dst];
      uint64_t dst_index = 0;
      uint64_t src_index = 0;

     igraph_adjlist_init(&graph, &al, IGRAPH_ALL, IGRAPH_NO_LOOPS, IGRAPH_NO_MULTIPLE);

      for (igraph_integer_t i = 0; i < total_src - 1; i++) {
	igraph_vs_all()
	igraph_vector_int_t *v_list = igraph_adjlist_get(&al, i);
	
printf("v_list: ",v_list);
	src[src_index] = dst_index;

	for (igraph_integer_t j = 0; j < igraph_vector_int_size(v_list); j++) {
	  dst[dst_index] = VECTOR(*v_list)[j];
	  dst_index++;
	}
	src_index ++;
	
									      
      }
      src[src_index] = dst_index;		     
      // broadcast graph
    printf("my_rank: %d before broadcast nprocs: %d no_of_nodes: %d edges: %d\n", my_rank, nprocs, no_of_nodes, total_dst);
    */
  }
  // broadcast graph
  /*
  MPI_Bcast(&total_src, 1, MPI_UINT64_T, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
   MPI_Bcast(&total_dst, 1, MPI_UINT64_T, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("my_rank: %d nprocs: %d no_of_nodes: %*" PRIu64 " edges: %*" PRIu64 "\n", my_rank, nprocs, 10, total_src, 10, total_dst);
  if (my_rank != ROOT) {
    src = new uint64_t[total_src];
    dst = new uint64_t[total_dst];
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(src, total_src, MPI_UINT64_T, ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(dst, total_dst, MPI_UINT64_T, ROOT, MPI_COMM_WORLD);
  

  MPI_Barrier(MPI_COMM_WORLD);
  my_work = no_of_nodes/nprocs;
  if (no_of_nodes%nprocs != 0)
    my_work++;



  nVVBuf[my_rank] = 1;
  nVV = 1;
  uint64_t iteration = 0;
  
  
  printf("my_rank: %d after bcast graph nprocs: %*" PRIu64 " my_work: %*" PRIu64 " vertices: %*" PRIu64 " edges: %*" PRIu64 "\n", my_rank, 2, nprocs, 10, my_work, 10, total_src, 10, total_dst);
  
  gettimeofday(&timecheck, NULL);
  astar_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
  int currLevel = 1;


  int compare = 1;
  
  uint64_t * buffer_open_recv = new uint64_t[my_work*nprocs];
  uint64_t * buffer_closed_recv = new uint64_t[my_work*nprocs];
  uint64_t * buffer_g_recv = new uint64_t[my_work*nprocs];
  uint64_t * buffer_h_recv = new uint64_t[my_work*nprocs];
  uint64_t * buffer_parent_recv = new uint64_t[my_work*nprocs];
  uint64_t * checkBuf = new uint64_t[nprocs];
  printf("my_rank: %d openList\n", my_rank);
  //output_graph(openList, grid_size, my_rank);
  printf("my_rank: %d closedList\n", my_rank);
  //output_graph(closedList, grid_size, my_rank);
    printf("my_rank: %d gList\n", my_rank);
    //output_graph(gList, grid_size, my_rank);
    printf("my_rank: %d nVVBuf\n", my_rank);
    //output_vec(nVVBuf, nprocs, my_rank);
    printf("my_rank: %d nVV: %*" PRIu64 "\n", my_rank, 10, nVV);
    
  MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD);
      printf("my_rank: %d checkBuf\n", my_rank);
      //output_vec(checkBuf, nprocs, my_rank);
  while (checkNVV(checkBuf, nprocs)) {
    // reset flags
 
    nVVBuf[my_rank] = 0;
    nVV = 0;

    //printf("inside while, my_rank: %d\n", my_rank);
    //    if (my_rank == 1)
    //	output_graph(openList, grid_size, my_rank);
    
    astar(openList, closedList, src, dst, gList, hList, parent, nVVBuf,nVV,  my_rank, my_work);
    
    
    // printf("my_rank: %d, condition: %d\n", my_rank, nVVBuf[my_rank]);
    // compare -- all processes are in loop
    //output_vec(checkBuf, nprocs, my_rank);
    // printf("my_rank: %d openList\n", my_rank);
    //output_graph(openList, grid_size, my_rank);
    //printf("my_rank: %d closedList\n", my_rank);
    //output_graph(closedList, grid_size, my_rank);
    //printf("my_rank: %d gList\n", my_rank);
      MPI_Barrier(MPI_COMM_WORLD);
      //printf("my_rank: %d nVVBuf\n" , my_rank);
      //    output_vec(nVVBuf, nprocs, my_rank);
    MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD);
    //printf("my_rank: %d checkBuf\n" , my_rank);
    
        MPI_Barrier(MPI_COMM_WORLD);
	//	output_vec(checkBuf, nprocs, my_rank);
    if (currLevel%compare == 0) {
      //printf("my_rank: %d compare\n", my_rank);

      MPI_Alltoall(gList, my_work, MPI_UINT64_T, buffer_g_recv, my_work, MPI_UINT64_T, world);
      MPI_Alltoall(hList, my_work, MPI_UINT64_T, buffer_h_recv, my_work, MPI_UINT64_T, world);
      MPI_Alltoall(openList, my_work, MPI_UINT64_T, buffer_open_recv, my_work, MPI_UINT64_T, world);
      MPI_Alltoall(closedList, my_work, MPI_UINT64_T, buffer_closed_recv, my_work, MPI_UINT64_T, world);
      MPI_Alltoall(parent, my_work, MPI_UINT64_T, buffer_parent_recv, my_work, MPI_UINT64_T, world);
          MPI_Barrier(MPI_COMM_WORLD);
      //printf("my_rank: %d after mpi alltoall\n", my_rank);
      //output_vec(buffer_level_recv, my_work*nprocs, my_rank);
	  

      //printf("my_rank: %d buffer_openList\n", my_rank);
      //output_graph(buffer_g_recv, grid_size, my_rank);
	  //  output_graph(buffer_open_recv, grid_size, my_rank);
      //printf("my_rank: %d buffer_closedList\n", my_rank);
      // output_graph(buffer_closed_recv, grid_size, my_rank);
      //printf("my_rank: %d buffer_gList\n", my_rank);
      //output_graph(buffer_g_recv, grid_size, my_rank);
      aggregate(buffer_g_recv, gList, my_rank, my_work,nprocs);
      aggregate(buffer_h_recv, hList, my_rank, my_work, nprocs);
      aggregate_open_or_closed(buffer_open_recv, openList, my_rank, my_work, nprocs);
      aggregate_open_or_closed(buffer_closed_recv, closedList, my_rank, my_work, nprocs);
      aggregate_parent(buffer_parent_recv, parent, gList, my_rank, my_work, nprocs);
          MPI_Barrier(MPI_COMM_WORLD);
	  //printf("my_rank: %d openList\n", my_rank);
	  //if (my_rank == 1)
	  //output_graph(gList, grid_size, my_rank);
	  //printf("my_rank: %d closedList\n", my_rank);
      //output_graph(closedList, grid_size, my_rank);
      //printf("my_rank: %d gList\n", my_rank);
	  //	  output_graph(parent, grid_size, my_rank);
      //printf("my_rank: %d after mpi aggregate\n", my_rank);      
      // output_vec(levelBuf, no_of_nodes, my_rank);
      //printf("my_rank: %d nVV flag: %d\n", my_rank, checkNVV(nprocs));
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD);
      //printf("my_rank: %d after all gather nVV flag: %d\n", my_rank, checkNVV(nprocs));
      }
    iteration ++;
  }

   MPI_Allgather(&parent[my_rank*my_work], my_work, MPI_UINT64_T, buffer_parent_recv, my_work, MPI_UINT64_T, MPI_COMM_WORLD);
   MPI_Allgather(&gList[my_rank*my_work], my_work, MPI_UINT64_T, buffer_g_recv, my_work, MPI_UINT64_T, MPI_COMM_WORLD);
   //     if (my_rank == 1)
   //     output_graph(gList, grid_size, my_rank);
  if (my_rank == end/my_work) {

    if (gList[end] != numeric_limits<uint64_t>::max()) {
      vector<uint64_t> path;
       uint64_t current = end;
       while (current != start) {
	 path.push_back(current);
	 current = buffer_parent_recv[current];
	 
       }
       path.push_back(current);
       uint64_t * path_arr = &path[0];
       output_vec(path_arr, path.size(), my_rank);
       printf("bfs_result: path exists with length: %*" PRIu64 "\n", 10, gList[end]);
    }
    else
      printf("bfs_result: path dne\n");
    gettimeofday(&timecheck, NULL);  
    astar_end = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;
    astar_elapsed = astar_end - astar_start;

    
    printf("***********************\n");
    printf("nodes: %d nprocs: %d time: %ld msecs\n", no_of_nodes, nprocs, astar_elapsed);
    }*/
  MPI_Finalize();




  return 0;

}
