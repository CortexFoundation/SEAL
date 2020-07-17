#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace seal;

class MNIST{
  public:
    void conv_batch(vector<vector<int64_t>>& images, vector<vector<int64_t>>& kernels, vector<vector<int64_t>>& outs, int batchs, int IC, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
      int OH = (H + 2*pad_h - KH) / stride + 1;
      int OW = (W + 2*pad_w - KW) / stride + 1;
      for(int b = 0; b < batchs; b++){
        for(int oc = 0; oc < out_channels; oc++){
          for(int oh = 0; oh < OH; oh++){
            for(int ow = 0; ow < OW; ow++){
              for(int ic = 0; ic < IC; ic++){
                int64_t sum = 0;
                for(int kh = 0; kh < KH; kh++){
                  for(int kw = 0; kw < KW; kw++){
                    int ih = oh * stride + kh - pad_h;
                    int iw = ow * stride + kw - pad_w;
                    if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                    }else{
                      sum += images[b][ic*H*W + ih*W + iw] * kernels[b][oc*KH*KW + kh*KW + kw];  
                    } 
                  }
                }
                //outs[b*out_channels*OH*OW + oc*OH*OW + oh*OW + ow] = sum;
                outs[b][(oc*IC + ic)*OH*OW + oh*OW + ow] = sum;
              }
            }
          }
        } 
      }
    }
    void square_activation(vector<vector<int64_t>>& data){
      for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
          int a = data[i][j];
          data[i][j] = a*a;
        }
      }
    }
    void scaled_mean_pool(vector<vector<int64_t>>& input, vector<vector<int64_t>>& output,
        int batchs, int H, int W, int KH, int KW, int padding, int out_channels){
      int OH = H + 2 * padding - KH + 1;
      int OW = W + 2 * padding - KW + 1;
      for(int b = 0; b < batchs; b++){
        for(int oc = 0; oc < out_channels; oc++){
          for(int oh = 0; oh < OH; oh++){
            for(int ow = 0; ow < OW; ow++){
              int sum = 0;
              for(int kh = 0; kh < KH; kh++){
                for(int kw = 0; kw < KW; kw++){
                  int ih = oh + kh - padding;
                  int iw = ow + kw - padding;
                  if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                  }else{
                    sum += input[b][oc*H*W + ih*W + iw];
                  }
                }
              } 
              output[b][oc *OH*OW + oh*OW + ow] = sum;
            }
          } 
        }
      }
    }
    void fully_connected(vector<vector<int64_t>>& weight, vector<vector<int64_t>>& input, vector<vector<int64_t>>& output, int batchs, int H, int W){
      for(int b = 0; b < batchs; b++){
        for(int i = 0; i < H; i++){
          int sum = 0;
          for(int j = 0; j < W; j++){
            int tmp = weight[b][i*W+j] * input[b][j];
            sum += tmp;
            //if(b == 1 && i == 0)
            //  cout << weight[b][i*W+j] << "," << input[b][j] << ", "  << tmp << "," << sum << endl;
          } 
          output[b][i] = sum;
        } 
      }
    }
};

#endif
