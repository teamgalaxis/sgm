/**
	This file is used as an interface to the sgm by Daniel Hernandez Juarez et. al.
	
	This c++ wrapper class is however, not written by this group but follows the 
	same licensing. See main.cu and other files.

**/
#pragma once
#ifndef SGM_API_H
#define SGM_API_H


#include <iostream>
#include <vector>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>




// Settings are set in configuration.h
typedef uint32_t cost_t;

class sgm {

public:
  sgm(uint8_t, uint8_t);
  void calcDisparity(cv::Mat im0, cv::Mat im1, cv::Mat &disp);

private:
  static cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
  static uint8_t *d_im0;
  static uint8_t *d_im1;
  static cost_t *d_transform0;
  static cost_t *d_transform1;
  static uint8_t *d_cost;
  static uint8_t *d_disparity;
  static uint8_t *d_disparity_filtered_uchar;
  static uint8_t *h_disparity;
  static uint16_t *d_S;
  static uint8_t *d_L0;
  static uint8_t *d_L1;
  static uint8_t *d_L2;
  static uint8_t *d_L3;
  static uint8_t *d_L4;
  static uint8_t *d_L5;
  static uint8_t *d_L6;
  static uint8_t *d_L7;
  static uint8_t p1, p2;
  static bool first_alloc;
  static uint32_t cols, rows, size, size_cube_l;

  float *elapsed_time_ms;
  
};


#endif
