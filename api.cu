#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include "debug.h"
#include "api.h"

cudaStream_t sgm::stream1, sgm::stream2, sgm::stream3;//, stream4, stream5, stream6, stream7, stream8;
uint8_t *sgm::d_im0;
uint8_t *sgm::d_im1;
cost_t *sgm::d_transform0;
cost_t *sgm::d_transform1;
uint8_t *sgm::d_cost;
uint8_t *sgm::d_disparity;
uint8_t *sgm::d_disparity_filtered_uchar;
uint8_t *sgm::h_disparity;
uint16_t *sgm::d_S;
uint8_t *sgm::d_L0;
uint8_t *sgm::d_L1;
uint8_t *sgm::d_L2;
uint8_t *sgm::d_L3;
uint8_t *sgm::d_L4;
uint8_t *sgm::d_L5;
uint8_t *sgm::d_L6;
uint8_t *sgm::d_L7;
uint8_t sgm::p1, sgm::p2;
bool sgm::first_alloc;
uint32_t sgm::cols, sgm::rows, sgm::size, sgm::size_cube_l;

sgm::sgm(uint8_t _p1, uint8_t _p2) {
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
  first_alloc = true;
  rows = 0;
  cols = 0;

  p1 = _p1;
  p2 = _p2;
}

void sgm::calcDisparity(cv::Mat im0, cv::Mat im1, cv::Mat &disp) {
  assert(im0.cols == im1.cols && im0.rows == im1.rows);
  assert(im0.type() == im1.type() && im0.channels() == im1.channels());
  assert(im0.cols % 4 == 0 && im0.rows % 4 == 0);
  assert(im0.channels() == 1 && im1.channels() == 1);
  
  cols = im0.cols;
  rows = im0.rows;
  size = cols*rows;
  size_cube_l = size*MAX_DISPARITY;
  if (first_alloc) {
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_transform0, sizeof(cost_t) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_transform1, sizeof(cost_t) * size));
  }
  int size_cube = size*MAX_DISPARITY;
  if(first_alloc) {
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_cost, sizeof(uint8_t) * size_cube));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_im0, sizeof(uint8_t) * size));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_im1, sizeof(uint8_t) * size));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_S, sizeof(uint16_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L0, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L1, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L2, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L3, sizeof(uint8_t) * size_cube_l));

#if PATH_AGGREGATION == 8
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
#endif

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_disparity, sizeof(uint8_t) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_disparity_filtered_uchar, sizeof(uint8_t) * size));
  }
  first_alloc = false;
  h_disparity = new uint8_t[size];

  CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, im0.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, im1.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 block_size;
  block_size.x = 32;
  block_size.y = 32;

  dim3 grid_size;
  grid_size.x = (cols+block_size.x-1) / block_size.x;
  grid_size.y = (rows+block_size.y+1) / block_size.y;

  CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

  // Hamming distance
  CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
  HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);

  // Cost aggregation
  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

  CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

#if PATH_AGGREGATION == 8
  CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

  CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
#endif  
  MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

  //cudaEventRecord(stop, 0);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  //cudaEventElapsedTime(elapsed_time_ms, start, stop);
  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

  disp = cv::Mat(rows,cols,CV_8UC1,h_disparity);

  /*
  CUDA_CHECK_RETURN(cudaFree(d_transform0));
  CUDA_CHECK_RETURN(cudaFree(d_transform1));
  CUDA_CHECK_RETURN(cudaFree(d_cost));
  CUDA_CHECK_RETURN(cudaFree(d_im0));
  CUDA_CHECK_RETURN(cudaFree(d_im1));
  CUDA_CHECK_RETURN(cudaFree(d_S));
  CUDA_CHECK_RETURN(cudaFree(d_L0));
  CUDA_CHECK_RETURN(cudaFree(d_L1));
  CUDA_CHECK_RETURN(cudaFree(d_L2));
  CUDA_CHECK_RETURN(cudaFree(d_L3));

#if PATH_AGGREGATION == 8
  CUDA_CHECK_RETURN(cudaFree(d_L4));
  CUDA_CHECK_RETURN(cudaFree(d_L5));
  CUDA_CHECK_RETURN(cudaFree(d_L6));
  CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif

  CUDA_CHECK_RETURN(cudaFree(d_disparity));
  CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
*/
}
