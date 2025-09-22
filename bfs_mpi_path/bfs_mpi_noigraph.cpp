/*
  bfs mpi no cuda path
  compile: mpicxx bfs_mpi_noigraph.cpp -o bfs_mpi_noig
  run: mpirun -n 2 bfs_mpi
 */
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <climits>
#include <vector>
#include <array>

#include <sys/time.h>
#include <random>
#include <cinttypes>

using namespace std;
using std::cerr;
using std::cout;
using std::endl;

#define MASTER 0
#define ROOT 0
/*
  nVVBuf(local) - 0 for no update, 1 for update
  checkBuf(global) - shared nVV
  srcPtrs, dst, srcPtrs_size, dst_size -> graph
  levelBuf - iteration level for vertex
 */
#define MAX_BUF_SIZE 1 << 25
uint64_t nVVBuf[MAX_BUF_SIZE], levelBuf[MAX_BUF_SIZE];
uint64_t checkBuf[MAX_BUF_SIZE];

uint64_t my_rankOps[MAX_BUF_SIZE];
uint64_t iterationOps[MAX_BUF_SIZE];
uint64_t messageOps[MAX_BUF_SIZE];
double totalTimeOps[MAX_BUF_SIZE];
uint64_t totalBytesOps[MAX_BUF_SIZE];

template <typename T> void output_vec(T *data, int data_size, int my_rank)
{
  printf("my_rank: %d: ", my_rank);
  for (int i = 0; i < data_size; i++)
    cout << data[i] << " ";
  printf("\n");
}

void output_graph(uint64_t *data, uint64_t grid_size, int my_rank)
{
  printf("my_rank: %d \n", my_rank);
  for (int i = 0; i < grid_size * grid_size; i++)
  {
    if (i % grid_size == 0)
      printf("\n");
    printf("%*" PRIu64 " ", 5, data[i]);
  }
  printf("\n");
}

bool checkNVV(uint64_t *checkBuf, int nprocs)
{
  for (int i = 0; i < nprocs; i++)
  {
    if (checkBuf[i] == 1)
      return true;
  }
  return false;
}
// for open and closed
void aggregate_open_or_closed(uint64_t *buffer_recv, uint64_t *buffer, int my_rank, int my_work, int nprocs)
{
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++)
  {
    for (uint64_t j = 0; j < nprocs; j++)
    {
      uint64_t buffer_recv_index = i + j * my_work;
      uint64_t buffer_index = i + my_rank * my_work;
      if (buffer_recv[buffer_recv_index] == 1 && buffer[buffer_index] == 0)
      {
        buffer[buffer_index] = 1;
        nVVBuf[my_rank] = 1;
      }
    }
  }
}
// for g List
void aggregate(uint64_t *buffer_recv, uint64_t *buffer, int my_rank, int my_work, int nprocs)
{
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++)
  {
    for (uint64_t j = 0; j < nprocs; j++)
    {
      uint64_t buffer_recv_index = i + j * my_work;
      uint64_t buffer_index = i + my_rank * my_work;
      if (buffer_recv[buffer_recv_index] != 0 && buffer_recv[i + j * my_work] < buffer[i + my_rank * my_work])
      {
        buffer[i + my_rank * my_work] = buffer_recv[i + j * my_work];
        //	printf("my_rank: %d buffer_index: %*" PRIu64 " %*" PRIu64 "\n", my_rank,10, buffer_index, 10, buffer_recv_index );
        nVVBuf[my_rank] = 1;
        //	printf("my_rank: %d levelBuf: %d bufflevelrecv: %d nVV: %d\n", my_rank, i+my_rank*my_work, j*my_work, nVVBuf[my_rank]);
      }
    }
  }
}
// for parent
void aggregate_parent(uint64_t *buffer_recv, uint64_t *buffer, uint64_t *gList, int my_rank, int my_work, int nprocs)
{
  uint64_t buffer_recv_index;
  uint64_t buffer_index;
  for (uint64_t i = 0; i < my_work; i++)
  {
    for (uint64_t j = 0; j < nprocs; j++)
    {
      uint64_t buffer_recv_index = i + j * my_work;
      uint64_t buffer_index = i + my_rank * my_work;
      if (buffer_recv[buffer_recv_index] != buffer[buffer_index] && gList[buffer_recv[buffer_recv_index]] < gList[buffer[buffer_index]])
      {
        buffer[buffer_index] = buffer_recv[buffer_recv_index];
        //	printf("my_rank: %d buffer_index: %*" PRIu64 " %*" PRIu64 "\n", my_rank,10, buffer_index, 10, buffer_recv_index );
        nVVBuf[my_rank] = 1;
        //	printf("my_rank: %d levelBuf: %d bufflevelrecv: %d nVV: %d\n", my_rank, i+my_rank*my_work, j*my_work, nVVBuf[my_rank]);
      }
    }
  }
  // if (buffer[15] != numeric_limits<uint64_t>::max()) {
  //   printf("my_rank: %d buffer_index: %*" PRIu64 " %*" PRIu64 "\n", my_rank,10, buffer_index, 10, buffer_recv_index );

  // }
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int emitter_rank, MPI_Comm communicator, int my_rank, int iteration, int message_id, uint64_t* opCount)
{
  double tstart = MPI_Wtime();
  int size;
  int result = PMPI_Bcast(buffer, count, datatype, emitter_rank, communicator);
  double totalTime = MPI_Wtime() - tstart;
  MPI_Type_size(datatype, &size);
  my_rankOps[opCount[my_rank]] = my_rank;
  iterationOps[opCount[my_rank]] = iteration;
  messageOps[opCount[my_rank]] = message_id;
  totalTimeOps[opCount[my_rank]] = totalTime;
  totalBytesOps[opCount[my_rank]] = count * size;
  opCount[my_rank]++;
  return result;
}
int MPI_Allgather(const void *buffer_send, int count_send, MPI_Datatype datatype_send, void *buffer_recv, int count_recv, MPI_Datatype datatype_recv, MPI_Comm communicator,int my_rank, int iteration, int message_id, uint64_t* opCount)
{
  double tstart = MPI_Wtime();
  int size;
  int result = PMPI_Allgather(buffer_send, count_send, datatype_send, buffer_recv, count_recv, datatype_recv, communicator);
  double totalTime = MPI_Wtime() - tstart;
  MPI_Type_size(datatype_send, &size);
  my_rankOps[opCount[my_rank]] = my_rank;
  iterationOps[opCount[my_rank]] = iteration;
  messageOps[opCount[my_rank]] = message_id;
  totalTimeOps[opCount[my_rank]] = totalTime;
  totalBytesOps[opCount[my_rank]] = count_send * size;
  opCount[my_rank]++;
  return result;
}
int MPI_Alltoall(const void *buffer_send, int count_send, MPI_Datatype datatype_send, void *buffer_recv, int count_recv, MPI_Datatype datatype_recv, MPI_Comm communicator,int my_rank, int iteration, int message_id, uint64_t* opCount)
{
  double tstart = MPI_Wtime();
  int size;
  int result = PMPI_Alltoall(buffer_send, count_send, datatype_send, buffer_recv, count_recv, datatype_recv, communicator);
  double totalTime = MPI_Wtime() - tstart;
  MPI_Type_size(datatype_send, &size);
  my_rankOps[opCount[my_rank]] = my_rank;
  iterationOps[opCount[my_rank]] = iteration;
  messageOps[opCount[my_rank]] = message_id;
  totalTimeOps[opCount[my_rank]] = totalTime;
  totalBytesOps[opCount[my_rank]] = count_send * size;
  opCount[my_rank]++;
  return result;
}

int MPI_Finalize(int my_rank, int nprocs, uint64_t * opCount)
{
  string opInfoArr[14] = {"MPI_Bcast total_src array", "MPI_Bcast total_dst array", "MPI_Bcast src array", "MPI_Bcast dst array", "MPI_Allgather nVV Visited before loop", "MPI_Allgather nVV Visited inside loop after bfs", "MPI_AlltoAll gList", "MPI_AlltoAll openList", "MPI_AlltoAll closedList", "MPI_AlltoAll parent", "MPI_Allgather nVV Visited inside loop after aggregate", "MPI_Allgather parent", "MPI_Allgather gList", "MPI_Allgather opCountArr"};
  // aggregate wall time, bytes transferred, op type
  uint64_t opCountArr[nprocs];
  MPI_Allgather(&opCount[my_rank], 1, MPI_UINT64_T, opCountArr, 1, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, -1, 13, opCount);
  int max_ops = 0;
  for (int i = 0; i < nprocs; i++) {
    if (opCountArr[i] > max_ops)
    max_ops = opCountArr[i];
  }

  uint64_t my_rankOpsAgg[max_ops*nprocs];
  int iterationOpsAgg[max_ops*nprocs];
  uint64_t messageOpsAgg[max_ops*nprocs];
  double totalTimeOpsAgg[max_ops*nprocs];
  uint64_t totalBytesOpsAgg[max_ops*nprocs];

  MPI_Gather(my_rankOps, max_ops, MPI_UINT64_T, my_rankOpsAgg, max_ops, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Gather(iterationOps, max_ops, MPI_INT, iterationOpsAgg, max_ops, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(messageOps, max_ops, MPI_UINT64_T, messageOpsAgg, max_ops, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Gather(totalTimeOps, max_ops, MPI_DOUBLE, totalTimeOpsAgg, max_ops, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(totalBytesOps, max_ops, MPI_UINT64_T, totalBytesOpsAgg, max_ops, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  ofstream bfs_mpi_noig_telemetry;
  double totalMpiTime[3];
  uint64_t totalBytesTransferredTime[3];
  bfs_mpi_noig_telemetry.open("../logs/bfs_mpi_noig_telemetry.txt") ;
  int log_index;
  string log_line;
  // output_vec(totalTimeOpsAgg, max_ops*nprocs, 0);
  if (my_rank == ROOT) {
    for (int i = 0; i < nprocs; i ++) {
      for (int j = 0; j < opCountArr[i]; j++) {
        log_index = i*max_ops+j;
        log_line = "my_rank: " + to_string(my_rankOpsAgg[log_index]) + " iteration: " + to_string(iterationOpsAgg[log_index]) + " MPI_Routine: " + opInfoArr[messageOpsAgg[log_index]] + " WallTime: " + to_string(totalTimeOpsAgg[log_index]) + " Bytes Transferred: " + to_string(totalBytesOpsAgg[log_index]) + "\n";
        bfs_mpi_noig_telemetry << log_line;
        if (messageOpsAgg[log_index] >= 0 && messageOpsAgg[log_index] < 4 ) {
          totalMpiTime[0] += totalTimeOpsAgg[log_index];
          totalBytesTransferredTime[0] += totalBytesOpsAgg[log_index];
        }
        else if (messageOpsAgg[log_index] == 4 || messageOpsAgg[log_index] == 5 && messageOpsAgg[log_index] >= 10) {
          totalMpiTime[1] += totalTimeOpsAgg[log_index];
          totalBytesTransferredTime[1] += totalBytesOpsAgg[log_index];
        } else if (messageOpsAgg[log_index] >= 6 && messageOpsAgg[log_index] <= 9){
          totalMpiTime[2] += totalTimeOpsAgg[log_index];
          totalBytesTransferredTime[2] += totalBytesOpsAgg[log_index];
        }
      }
    }
  }
  bfs_mpi_noig_telemetry << "---------------------Cumulative Time and Bytes Transferred by Routine Type--------------------\n";
  bfs_mpi_noig_telemetry << "Wall Time\n";
  log_line = "MPI_Bcast: " + to_string(totalMpiTime[0]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  log_line = "MPI_Allgather: " + to_string(totalMpiTime[1]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  log_line = "MPI_Alltoall: " + to_string(totalMpiTime[2]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  bfs_mpi_noig_telemetry << "Bytes Transferred\n";
  log_line = "MPI_Bcast: " + to_string(totalBytesTransferredTime[0]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  log_line = "MPI_Allgather: " + to_string(totalBytesTransferredTime[1]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  log_line = "MPI_Alltoall: " + to_string(totalBytesTransferredTime[2]) + "\n";
  bfs_mpi_noig_telemetry << log_line;
  bfs_mpi_noig_telemetry.close();
  int result = PMPI_Finalize();
  return result;
}
// MPI_Barrier



void bfs(uint64_t *openList, uint64_t *closedList, uint64_t *src, uint64_t *dst, uint64_t *gList, uint64_t *parent, uint64_t *nVVBuf, uint64_t nVV, int my_rank, uint64_t my_work)
{
  uint64_t g_score;
  // printf("my_rank: %d inside astar\n", my_rank);
  /* if (my_rank == 1) {
      output_graph(gList, 6, my_rank);
  output_graph(openList, 6, my_rank);
  }*/

  for (int i = 0; i < my_work; i++)
  {
    uint64_t index = my_rank * my_work + i;

    if (openList[index] == 1 && closedList[index] == 0)
    {
      // printf("my_rank: %d index: %*" PRIu64 "\n", my_rank, 10, index);
      nVVBuf[my_rank] = 1;
      nVV = 1;
      openList[index] = 0;
      closedList[index] = 1;
      uint64_t start = src[index];
      uint64_t end = src[index + 1];
      for (uint64_t j = start; j < end; j++)
      {

        uint64_t nb = dst[j];
        // if (index == 46 || index == 17772)
        //   printf("my_rank: %d index: %d neighbors: %d\n", my_rank, index, nb);
        if (closedList[nb] == 1)
          continue;
        g_score = gList[index] + 1;
        // printf("my_rank: %d nb: %*" PRIu64 "\n", my_rank, 10, nb);

        if (openList[nb] == 0)
          openList[nb] = 1;
        else if (gList[nb] != 0 && g_score >= gList[nb])
          continue;
        gList[nb] = g_score;
        parent[nb] = index;

        // printf("my_rank: %d nb: %*" PRIu64 " parent: %*" PRIu64 " g_score: %*" PRIu64 "\n",my_rank, 10, nb, 10, index, 10, g_score);
      }
    }
  }
};

int main(int argc, char *argv[])
{
  int my_rank, nprocs;
  uint64_t my_work;
  uint64_t no_of_nodes;
  uint64_t grid_size;

  uint64_t start, end;

  uint64_t *src;
  uint64_t *dst;
  uint64_t total_src;
  uint64_t total_dst;

  long bfs_start, bfs_end, bfs_elapsed;

  struct timeval timecheck;

  uint64_t nVV;

  int astar_result = -1;

  MPI_Comm world = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // Initialization
  if (argc == 3)
  {
    int i = 1;
    start = atoi(argv[i++]);
    end = atoi(argv[i++]);
  }
  else
  {
    start = 0;
    end = 999;
  }

  printf("my_rank: %d start: %d, end: %d\n", my_rank, start, end);
  if (my_rank == ROOT)
  {
    // Read csr format

    printf("citation read begins\n");

    // FILE *input;

    ifstream input("../datasets/dataset_mapped_csr.txt");
    string readline;
    int count = 0;
    int index = 0;
    while (getline(input, readline))
    {
      if (count == 0)
      {
        total_src = stoi(readline);
        src = new uint64_t[total_src];
        count++;
      }
      else if (count == 1)
      {
        total_dst = stoi(readline);
        dst = new uint64_t[total_dst];
        count++;
      }
      else if (count == 2)
      {
        if (readline == "")
        {
          count++;
          index = 0;
        }
        else
        {
          src[index] = stoi(readline);
          index++;
        }
      }
      else if (count == 3)
      {
        dst[index] = stoi(readline);
        index++;
      }
      else
      {
        printf("Error in dataset_mapped_csr file");
      }
    }
    printf("my_rank: %d before broadcast nprocs: %d no_of_nodes: %d edges: %d\n", my_rank, nprocs, no_of_nodes, total_dst);
    // output_vec(src, 100,0);
    // output_vec(dst, 100,0);
  }
  uint64_t opCount[nprocs];
  nVVBuf[my_rank] = 1;
  nVV = 1;
  opCount[my_rank] = 0;

  // broadcast graph
  MPI_Bcast(&total_src, 1, MPI_UINT64_T, ROOT, MPI_COMM_WORLD, my_rank, -1, 0, opCount);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&total_dst, 1, MPI_UINT64_T, ROOT, MPI_COMM_WORLD, my_rank, -1, 1, opCount);
  MPI_Barrier(MPI_COMM_WORLD);
  no_of_nodes = total_src - 1;
  printf("my_rank: %d nprocs: %d no_of_nodes: %*" PRIu64 " edges: %*" PRIu64 "\n", my_rank, nprocs, 10, no_of_nodes, 10, total_dst);
  if (start >= no_of_nodes || end >= no_of_nodes)
  {
    printf("Error: start_node %d or end_node %d has to be valid node[0-%d]\n", start, end, no_of_nodes - 1);
  }
  my_work = no_of_nodes / nprocs;
  if (no_of_nodes % nprocs != 0)
    my_work++;

  uint64_t *gList = new uint64_t[no_of_nodes];
  uint64_t *openList = new uint64_t[no_of_nodes];
  uint64_t *closedList = new uint64_t[no_of_nodes];
  uint64_t *parent = new uint64_t[no_of_nodes];

  for (uint64_t i = 0; i < no_of_nodes; i++)
  {
    gList[i] = numeric_limits<uint64_t>::max();
    openList[i] = 0;
    closedList[i] = 0;
    parent[i] = numeric_limits<uint64_t>::max();
  }
  gList[start] = 0;

  openList[start] = 1;
  printf("arrays initialized\n");
  if (my_rank != ROOT)
  {
    src = new uint64_t[total_src];
    dst = new uint64_t[total_dst];
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(src, total_src, MPI_UINT64_T, ROOT, MPI_COMM_WORLD, my_rank, -1, 2, opCount);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(dst, total_dst, MPI_UINT64_T, ROOT, MPI_COMM_WORLD, my_rank, -1, 3, opCount);

  MPI_Barrier(MPI_COMM_WORLD);

  
  uint64_t iteration = 0;

  printf("my_rank: %d after bcast graph nprocs: %*" PRIu64 " my_work: %*" PRIu64 " vertices: %*" PRIu64 " edges: %*" PRIu64 "\n", my_rank, 2, nprocs, 10, my_work, 10, total_src, 10, total_dst);

  gettimeofday(&timecheck, NULL);
  bfs_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  int currLevel = 1;

  int compare = 1;

  uint64_t *buffer_open_recv = new uint64_t[my_work * nprocs];
  uint64_t *buffer_closed_recv = new uint64_t[my_work * nprocs];
  uint64_t *buffer_g_recv = new uint64_t[my_work * nprocs];
  uint64_t *buffer_h_recv = new uint64_t[my_work * nprocs];
  uint64_t *buffer_parent_recv = new uint64_t[my_work * nprocs];
  uint64_t *checkBuf = new uint64_t[nprocs];
  printf("my_rank: %d openList\n", my_rank);
  // output_graph(openList, grid_size, my_rank);
  printf("my_rank: %d closedList\n", my_rank);
  // output_graph(closedList, grid_size, my_rank);
  printf("my_rank: %d gList\n", my_rank);
  // output_graph(gList, grid_size, my_rank);
  printf("my_rank: %d nVVBuf\n", my_rank);
  // output_vec(nVVBuf, nprocs, my_rank);
  printf("my_rank: %d nVV: %*" PRIu64 "\n", my_rank, 10, nVV);

  MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, iteration, 4, opCount);
  printf("my_rank: %d checkBuf\n", my_rank);
  // output_vec(checkBuf, nprocs, my_rank);
  while (checkNVV(checkBuf, nprocs))
  {
    // reset flags

    nVVBuf[my_rank] = 0;
    nVV = 0;

    // printf("inside while, my_rank: %d\n", my_rank);
    //     if (my_rank == 1)
    //	output_graph(openList, grid_size, my_rank);

    bfs(openList, closedList, src, dst, gList, parent, nVVBuf, nVV, my_rank, my_work);

    // if (my_rank == 1)
    //   output_graph(gList, 5, my_rank);
    // printf("my_rank: %d, condition: %d\n", my_rank, nVVBuf[my_rank]);
    // compare -- all processes are in loop
    // output_vec(checkBuf, nprocs, my_rank);
    // printf("my_rank: %d openList\n", my_rank);
    // output_graph(openList, grid_size, my_rank);
    // printf("my_rank: %d closedList\n", my_rank);
    // output_graph(closedList, grid_size, my_rank);
    // printf("my_rank: %d gList\n", my_rank);
    /*if (my_rank == 1)
      output_graph(openList, grid_size, my_rank);
    */
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("my_rank: %d nVVBuf\n" , my_rank);
    //     output_vec(nVVBuf, nprocs, my_rank);
    MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, iteration, 5, opCount);
    // printf("my_rank: %d checkBuf\n" , my_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    //	output_vec(checkBuf, nprocs, my_rank);
    if (currLevel % compare == 0)
    {
      // printf("my_rank: %d compare\n", my_rank);

      MPI_Alltoall(gList, my_work, MPI_UINT64_T, buffer_g_recv, my_work, MPI_UINT64_T, world, my_rank, iteration, 6, opCount);
      MPI_Alltoall(openList, my_work, MPI_UINT64_T, buffer_open_recv, my_work, MPI_UINT64_T, world, my_rank, iteration, 7, opCount);
      MPI_Alltoall(closedList, my_work, MPI_UINT64_T, buffer_closed_recv, my_work, MPI_UINT64_T, world, my_rank, iteration, 8, opCount);
      MPI_Alltoall(parent, my_work, MPI_UINT64_T, buffer_parent_recv, my_work, MPI_UINT64_T, world, my_rank, iteration, 9, opCount);
      MPI_Barrier(MPI_COMM_WORLD);
      // printf("my_rank: %d after mpi alltoall\n", my_rank);
      // output_vec(buffer_level_recv, my_work*nprocs, my_rank);

      // printf("my_rank: %d buffer_openList\n", my_rank);
      // output_graph(buffer_g_recv, grid_size, my_rank);
      //  output_graph(buffer_open_recv, grid_size, my_rank);
      // printf("my_rank: %d buffer_closedList\n", my_rank);
      // output_graph(buffer_closed_recv, grid_size, my_rank);
      // printf("my_rank: %d buffer_gList\n", my_rank);
      // output_graph(buffer_g_recv, grid_size, my_rank);
      aggregate(buffer_g_recv, gList, my_rank, my_work, nprocs);
      aggregate_open_or_closed(buffer_open_recv, openList, my_rank, my_work, nprocs);
      aggregate_open_or_closed(buffer_closed_recv, closedList, my_rank, my_work, nprocs);
      aggregate_parent(buffer_parent_recv, parent, gList, my_rank, my_work, nprocs);
      MPI_Barrier(MPI_COMM_WORLD);
      // printf("my_rank: %d openList\n", my_rank);
      // if (my_rank == 1)
      // output_graph(gList, grid_size, my_rank);
      // printf("my_rank: %d closedList\n", my_rank);
      // output_graph(closedList, grid_size, my_rank);
      // printf("my_rank: %d gList\n", my_rank);
      //	  output_graph(parent, grid_size, my_rank);
      // printf("my_rank: %d after mpi aggregate\n", my_rank);
      // output_vec(levelBuf, no_of_nodes, my_rank);
      // printf("my_rank: %d nVV flag: %d\n", my_rank, checkNVV(nprocs));
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allgather(&nVVBuf[my_rank], 1, MPI_UINT64_T, checkBuf, 1, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, iteration, 10, opCount);
      // printf("my_rank: %d after all gather nVV flag: %d\n", my_rank, checkNVV(nprocs));
    }
    iteration++;
  }

  MPI_Allgather(&parent[my_rank * my_work], my_work, MPI_UINT64_T, buffer_parent_recv, my_work, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, iteration, 11, opCount);
  MPI_Allgather(&gList[my_rank * my_work], my_work, MPI_UINT64_T, buffer_g_recv, my_work, MPI_UINT64_T, MPI_COMM_WORLD, my_rank, iteration, 12, opCount);
  // if (my_rank == 1)
  // output_graph(gList, 5, my_rank);
  if (my_rank == end / my_work)
  {

    if (gList[end] != numeric_limits<uint64_t>::max())
    {
      vector<uint64_t> path;
      uint64_t current = end;
      while (current != start)
      {
        path.push_back(current);
        // printf("my_rank: %d current: %d\n", my_rank, current);
        current = parent[current];
      }
      path.push_back(current);
      uint64_t *path_arr = &path[0];
      output_vec(path_arr, path.size(), my_rank);
      printf("bfs_result: path exists with length: %*" PRIu64 "\n", 10, gList[end]);
    }
    else
      printf("bfs_result: path dne\n");
    gettimeofday(&timecheck, NULL);
    bfs_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    bfs_elapsed = bfs_end - bfs_start;

    printf("***********************\n");
    printf("nodes: %d nprocs: %d time: %ld msecs\n", no_of_nodes, nprocs, bfs_elapsed);
  }
  MPI_Finalize(my_rank, nprocs, opCount);

  return 0;
}
