#include "mnist_he.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace seal;


void rand_init_vector(vector<vector<uint64_t>>& data, int rows, int cols){
  for(int i = 0; i < rows; i++){
    data[i].resize(cols);
    for(int j = 0; j < data[i].size(); j++){
      data[i][j] = rand()%10;
    }
  }
}
void transpose_vector(vector<vector<uint64_t>>& in, vector<vector<uint64_t>>& out, int rows, int cols){
  for(int i = 0; i < cols; i++){
    vector<uint64_t> row_data(rows);
    for(int j = 0; j < rows; j++){
      row_data[j] = in[j][i];
    }
    out[i] = row_data;
  }
}

int main(){
  //init input data
  int batchs = 8192;
  int H = 28, W = 28;
  int KH = 5, KW = 5;
  int stride = 2;
  int pad_h = 1, pad_w = 1;
  int out_channels = 5;
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  vector<vector<uint64_t>> images(batchs), kernels(batchs), outs(batchs);//, out2(batchs);
  rand_init_vector(images, batchs, H*W);
  rand_init_vector(kernels, batchs, out_channels * KH * KW);

  for(int i = 0; i < batchs; i++){
    outs[i].resize(out_channels * OH * OW);
  }

  //
  MNIST mnist(8192, true);

  //encrypted input image and weight
  vector<Ciphertext> image_encrypted(H*W), kernel_encrypted(out_channels * KH*KW), out_encrypted(out_channels * OH * OW);
  double start = omp_get_wtime();
  vector<vector<uint64_t>> batch_images(H*W), batch_kernels(out_channels * KH*KW), batch_outs(out_channels*OH*OW);
  transpose_vector(images, batch_images, batchs, H*W);
  transpose_vector(kernels, batch_kernels, batchs, out_channels * KH * KW);
  for(int i = 0; i < out_channels*OH*OW; i++){
    batch_outs[i].resize(batchs);
  }
  mnist.encrypted(batch_images, image_encrypted);
  mnist.encrypted(batch_kernels, kernel_encrypted);
  double end = omp_get_wtime();
  cout << "the times of encrypted image and kernel: " << (double)(end-start) << "s" << endl;

  start = omp_get_wtime();
  //infer
  //conv
  mnist.conv_batch(image_encrypted, kernel_encrypted, out_encrypted, batchs, 1, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
  //activation
  mnist.square_activation(out_encrypted);
  //pool
  H = OH; W = OW;
  int kernel_size = 3, padding = 1;
  OH = H + 2*padding - kernel_size + 1; 
  OW = W + 2*padding - kernel_size + 1;
  vector<Ciphertext> pool_out(out_channels * OH *OW);
  mnist.scaled_mean_pool(out_encrypted, pool_out, batchs, H, W, kernel_size, kernel_size, padding, out_channels);
  //conv
  out_channels = 10;
  H = OH; W = OW; pad_h = 0; pad_w = 0; stride = 2;
  OH = (H + 2*pad_h - KH) / stride + 1;
  OW = (W + 2*pad_w - KW) / stride + 1;
  vector<vector<uint64_t>> kernels2(batchs), batch_kernels2(out_channels*KH*KW);
  vector<Ciphertext> kernel_encrypted2(out_channels * KH*KW);
  rand_init_vector(kernels2, batchs, out_channels*KH*KW);
  transpose_vector(kernels2, batch_kernels2, batchs, out_channels*KH*KW);
  mnist.encrypted(batch_kernels2, kernel_encrypted2);
  int IC = 5;
  vector<Ciphertext> conv_out2(IC*out_channels * OH * OW);
  mnist.conv_batch(pool_out, kernel_encrypted2, conv_out2, batchs, 5, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
  //pool
  vector<Ciphertext> pool_out2(IC*out_channels * OH * OW);
  mnist.scaled_mean_pool(conv_out2, pool_out2, batchs, OH, OW, kernel_size, kernel_size, padding, IC*out_channels);
  //fully connected
  int out_nodes = 100;
  cout << "pool out size = " << pool_out2.size() << endl;
  vector<vector<uint64_t>> fc_weight(batchs), batch_fc_weight(out_nodes*pool_out2.size()); 
  rand_init_vector(fc_weight, batchs, out_nodes*pool_out2.size());
  transpose_vector(fc_weight, batch_fc_weight, batchs, out_nodes*pool_out2.size());
  vector<Ciphertext> fc_weight_encrypted(out_nodes * pool_out2.size());
  //mnist.encrypted(batch_fc_weight, fc_weight_encrypted);
  vector<Ciphertext> fc_out(out_nodes);
  //mnist.fully_connected(fc_weight_encrypted, pool_out2, fc_out, batchs, out_nodes, pool_out2.size());
  mnist.fully_connected(batch_fc_weight, pool_out2, fc_out, batchs, out_nodes, pool_out2.size());
  //activation
  mnist.square_activation(fc_out);
  //fully connected
  out_nodes = 10;
  vector<vector<uint64_t>> fc_weight2(batchs), batch_fc_weight2(out_nodes*pool_out2.size()); 
  rand_init_vector(fc_weight2, batchs, out_nodes*pool_out2.size());
  transpose_vector(fc_weight2, batch_fc_weight2, batchs, out_nodes*pool_out2.size());
  vector<Ciphertext> fc_weight2_encrypted(out_nodes * fc_out.size());
//  mnist.encrypted(batch_fc_weight2, fc_weight2_encrypted);
  vector<Ciphertext> fc_out2(out_nodes);
 // mnist.fully_connected(fc_weight2_encrypted, fc_out, fc_out2, batchs, out_nodes, fc_out.size());
  mnist.fully_connected(batch_fc_weight2, fc_out, fc_out2, batchs, out_nodes, fc_out.size());

  end = omp_get_wtime();
  cout << "infer time = " << end-start << "s" << endl;

  start = omp_get_wtime();
  mnist.decrypted(fc_out2, outs);
  end = omp_get_wtime();
  cout << "decrypted result time = " << end-start << "s" << endl;
}
