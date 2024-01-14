#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16


__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ((W_out - 1) / TILE_WIDTH) + 1;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int height = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // # block in row
    int width = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // # block in col

    if (height < H_out && width < W_out) {
        float out = 0.0f;
        for (int c = 0; c < C; c++) {

            if (K == 1) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }

            }else if (K == 2) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }

            }else if (K == 3) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 0 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 2) * mask_4d(by, c, 0, 2);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }
                if (height * S + 1 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 2) * mask_4d(by, c, 1, 2);
                    }
                if (height * S + 2 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 0) * mask_4d(by, c, 2, 0);
                    }
                if (height * S + 2 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 1) * mask_4d(by, c, 2, 1);
                    }
                if (height * S + 2 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 2) * mask_4d(by, c, 2, 2);
                    }

            }else if (K == 4) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 0 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 2) * mask_4d(by, c, 0, 2);
                    }
                if (height * S + 0 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 3) * mask_4d(by, c, 0, 3);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }
                if (height * S + 1 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 2) * mask_4d(by, c, 1, 2);
                    }
                if (height * S + 1 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 3) * mask_4d(by, c, 1, 3);
                    }
                if (height * S + 2 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 0) * mask_4d(by, c, 2, 0);
                    }
                if (height * S + 2 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 1) * mask_4d(by, c, 2, 1);
                    }
                if (height * S + 2 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 2) * mask_4d(by, c, 2, 2);
                    }
                if (height * S + 2 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 3) * mask_4d(by, c, 2, 3);
                    }
                if (height * S + 3 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 0) * mask_4d(by, c, 3, 0);
                    }
                if (height * S + 3 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 1) * mask_4d(by, c, 3, 1);
                    }
                if (height * S + 3 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 2) * mask_4d(by, c, 3, 2);
                    }
                if (height * S + 3 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 3) * mask_4d(by, c, 3, 3);
                    }

            }else if (K == 5) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 0 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 2) * mask_4d(by, c, 0, 2);
                    }
                if (height * S + 0 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 3) * mask_4d(by, c, 0, 3);
                    }
                if (height * S + 0 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 4) * mask_4d(by, c, 0, 4);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }
                if (height * S + 1 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 2) * mask_4d(by, c, 1, 2);
                    }
                if (height * S + 1 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 3) * mask_4d(by, c, 1, 3);
                    }
                if (height * S + 1 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 4) * mask_4d(by, c, 1, 4);
                    }
                if (height * S + 2 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 0) * mask_4d(by, c, 2, 0);
                    }
                if (height * S + 2 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 1) * mask_4d(by, c, 2, 1);
                    }
                if (height * S + 2 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 2) * mask_4d(by, c, 2, 2);
                    }
                if (height * S + 2 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 3) * mask_4d(by, c, 2, 3);
                    }
                if (height * S + 2 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 4) * mask_4d(by, c, 2, 4);
                    }
                if (height * S + 3 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 0) * mask_4d(by, c, 3, 0);
                    }
                if (height * S + 3 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 1) * mask_4d(by, c, 3, 1);
                    }
                if (height * S + 3 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 2) * mask_4d(by, c, 3, 2);
                    }
                if (height * S + 3 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 3) * mask_4d(by, c, 3, 3);
                    }
                if (height * S + 3 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 4) * mask_4d(by, c, 3, 4);
                    }
                if (height * S + 4 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 0) * mask_4d(by, c, 4, 0);
                    }
                if (height * S + 4 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 1) * mask_4d(by, c, 4, 1);
                    }
                if (height * S + 4 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 2) * mask_4d(by, c, 4, 2);
                    }
                if (height * S + 4 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 3) * mask_4d(by, c, 4, 3);
                    }
                if (height * S + 4 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 4) * mask_4d(by, c, 4, 4);
                    }

            }else if (K == 6) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 0 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 2) * mask_4d(by, c, 0, 2);
                    }
                if (height * S + 0 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 3) * mask_4d(by, c, 0, 3);
                    }
                if (height * S + 0 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 4) * mask_4d(by, c, 0, 4);
                    }
                if (height * S + 0 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 5) * mask_4d(by, c, 0, 5);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }
                if (height * S + 1 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 2) * mask_4d(by, c, 1, 2);
                    }
                if (height * S + 1 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 3) * mask_4d(by, c, 1, 3);
                    }
                if (height * S + 1 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 4) * mask_4d(by, c, 1, 4);
                    }
                if (height * S + 1 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 5) * mask_4d(by, c, 1, 5);
                    }
                if (height * S + 2 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 0) * mask_4d(by, c, 2, 0);
                    }
                if (height * S + 2 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 1) * mask_4d(by, c, 2, 1);
                    }
                if (height * S + 2 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 2) * mask_4d(by, c, 2, 2);
                    }
                if (height * S + 2 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 3) * mask_4d(by, c, 2, 3);
                    }
                if (height * S + 2 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 4) * mask_4d(by, c, 2, 4);
                    }
                if (height * S + 2 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 5) * mask_4d(by, c, 2, 5);
                    }
                if (height * S + 3 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 0) * mask_4d(by, c, 3, 0);
                    }
                if (height * S + 3 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 1) * mask_4d(by, c, 3, 1);
                    }
                if (height * S + 3 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 2) * mask_4d(by, c, 3, 2);
                    }
                if (height * S + 3 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 3) * mask_4d(by, c, 3, 3);
                    }
                if (height * S + 3 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 4) * mask_4d(by, c, 3, 4);
                    }
                if (height * S + 3 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 5) * mask_4d(by, c, 3, 5);
                    }
                if (height * S + 4 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 0) * mask_4d(by, c, 4, 0);
                    }
                if (height * S + 4 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 1) * mask_4d(by, c, 4, 1);
                    }
                if (height * S + 4 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 2) * mask_4d(by, c, 4, 2);
                    }
                if (height * S + 4 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 3) * mask_4d(by, c, 4, 3);
                    }
                if (height * S + 4 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 4) * mask_4d(by, c, 4, 4);
                    }
                if (height * S + 4 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 5) * mask_4d(by, c, 4, 5);
                    }
                if (height * S + 5 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 0) * mask_4d(by, c, 5, 0);
                    }
                if (height * S + 5 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 1) * mask_4d(by, c, 5, 1);
                    }
                if (height * S + 5 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 2) * mask_4d(by, c, 5, 2);
                    }
                if (height * S + 5 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 3) * mask_4d(by, c, 5, 3);
                    }
                if (height * S + 5 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 4) * mask_4d(by, c, 5, 4);
                    }
                if (height * S + 5 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 5) * mask_4d(by, c, 5, 5);
                    }

            }else if (K == 7) {
                if (height * S + 0 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 0) * mask_4d(by, c, 0, 0);
                    }
                if (height * S + 0 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 1) * mask_4d(by, c, 0, 1);
                    }
                if (height * S + 0 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 2) * mask_4d(by, c, 0, 2);
                    }
                if (height * S + 0 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 3) * mask_4d(by, c, 0, 3);
                    }
                if (height * S + 0 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 4) * mask_4d(by, c, 0, 4);
                    }
                if (height * S + 0 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 5) * mask_4d(by, c, 0, 5);
                    }
                if (height * S + 0 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 0, width * S + 6) * mask_4d(by, c, 0, 6);
                    }
                if (height * S + 1 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 0) * mask_4d(by, c, 1, 0);
                    }
                if (height * S + 1 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 1) * mask_4d(by, c, 1, 1);
                    }
                if (height * S + 1 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 2) * mask_4d(by, c, 1, 2);
                    }
                if (height * S + 1 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 3) * mask_4d(by, c, 1, 3);
                    }
                if (height * S + 1 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 4) * mask_4d(by, c, 1, 4);
                    }
                if (height * S + 1 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 5) * mask_4d(by, c, 1, 5);
                    }
                if (height * S + 1 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 1, width * S + 6) * mask_4d(by, c, 1, 6);
                    }
                if (height * S + 2 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 0) * mask_4d(by, c, 2, 0);
                    }
                if (height * S + 2 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 1) * mask_4d(by, c, 2, 1);
                    }
                if (height * S + 2 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 2) * mask_4d(by, c, 2, 2);
                    }
                if (height * S + 2 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 3) * mask_4d(by, c, 2, 3);
                    }
                if (height * S + 2 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 4) * mask_4d(by, c, 2, 4);
                    }
                if (height * S + 2 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 5) * mask_4d(by, c, 2, 5);
                    }
                if (height * S + 2 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 2, width * S + 6) * mask_4d(by, c, 2, 6);
                    }
                if (height * S + 3 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 0) * mask_4d(by, c, 3, 0);
                    }
                if (height * S + 3 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 1) * mask_4d(by, c, 3, 1);
                    }
                if (height * S + 3 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 2) * mask_4d(by, c, 3, 2);
                    }
                if (height * S + 3 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 3) * mask_4d(by, c, 3, 3);
                    }
                if (height * S + 3 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 4) * mask_4d(by, c, 3, 4);
                    }
                if (height * S + 3 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 5) * mask_4d(by, c, 3, 5);
                    }
                if (height * S + 3 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 3, width * S + 6) * mask_4d(by, c, 3, 6);
                    }
                if (height * S + 4 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 0) * mask_4d(by, c, 4, 0);
                    }
                if (height * S + 4 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 1) * mask_4d(by, c, 4, 1);
                    }
                if (height * S + 4 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 2) * mask_4d(by, c, 4, 2);
                    }
                if (height * S + 4 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 3) * mask_4d(by, c, 4, 3);
                    }
                if (height * S + 4 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 4) * mask_4d(by, c, 4, 4);
                    }
                if (height * S + 4 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 5) * mask_4d(by, c, 4, 5);
                    }
                if (height * S + 4 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 4, width * S + 6) * mask_4d(by, c, 4, 6);
                    }
                if (height * S + 5 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 0) * mask_4d(by, c, 5, 0);
                    }
                if (height * S + 5 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 1) * mask_4d(by, c, 5, 1);
                    }
                if (height * S + 5 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 2) * mask_4d(by, c, 5, 2);
                    }
                if (height * S + 5 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 3) * mask_4d(by, c, 5, 3);
                    }
                if (height * S + 5 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 4) * mask_4d(by, c, 5, 4);
                    }
                if (height * S + 5 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 5) * mask_4d(by, c, 5, 5);
                    }
                if (height * S + 5 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 5, width * S + 6) * mask_4d(by, c, 5, 6);
                    }
                if (height * S + 6 < H && width * S + 0 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 0) * mask_4d(by, c, 6, 0);
                    }
                if (height * S + 6 < H && width * S + 1 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 1) * mask_4d(by, c, 6, 1);
                    }
                if (height * S + 6 < H && width * S + 2 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 2) * mask_4d(by, c, 6, 2);
                    }
                if (height * S + 6 < H && width * S + 3 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 3) * mask_4d(by, c, 6, 3);
                    }
                if (height * S + 6 < H && width * S + 4 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 4) * mask_4d(by, c, 6, 4);
                    }
                if (height * S + 6 < H && width * S + 5 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 5) * mask_4d(by, c, 6, 5);
                    }
                if (height * S + 6 < H && width * S + 6 < W) {
                    out += in_4d(bx, c, height * S + 6, width * S + 6) * mask_4d(by, c, 6, 6);
                    }

            }else{
                for (int p=0; p<K; p++) {
                    for (int q=0; q<K; q++) {
                        if (height * S + p < H && width * S + q < W) {
                            out += in_4d(bx, c, height * S + p, width * S + q) * mask_4d(by, c, p, q);
                        }
                    }
                }
            }
        }
        out_4d(bx, by, height, width) = out;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float * __restrict__ host_output, const float * __restrict__ host_input, const float * __restrict__ host_mask, float ** __restrict__ device_output_ptr, float ** __restrict__ device_input_ptr, float ** __restrict__ device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int in_size = B * C * H * W;
    int out_size = B * M * H_out * W_out;
    int mask_size = M * C * K * K;

    cudaMalloc((void**) device_input_ptr, in_size* sizeof(float));
    cudaMalloc((void**) device_output_ptr, out_size* sizeof(float));
    cudaMalloc((void**) device_mask_ptr, mask_size* sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, in_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size*sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float * __restrict__ device_output, const float * __restrict__ device_input, const float * __restrict__ device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int H_grid = ((H_out - 1)/ TILE_WIDTH) + 1;
    int W_grid = ((W_out - 1)/ TILE_WIDTH) + 1;
    int Z_gird = H_grid * W_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, Z_gird);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S); 

    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float * __restrict__ host_output, float * __restrict__ device_output, float * __restrict__ device_input, float * __restrict__ device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int out_size = B * M * H_out * W_out;

    cudaMemcpy(host_output, device_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);  
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}