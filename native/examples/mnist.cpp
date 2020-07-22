#include "mnist_he.hpp"
#include "mnist.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

using namespace std;
using namespace seal;


void rand_input(vector<vector<int64_t>>& data, int rows, int cols){
  for(int i = 0; i < rows; i++){
    data[i].resize(cols);
    for(int j = 0; j < cols; j++){
      data[i][j] = rand()%2;
    }
  }
}
void rand_weight(vector<vector<int64_t>>& data, int rows, int cols){
  for(int i = 0; i < rows; i++){
    data[i].resize(cols);
    for(int j = 0; j < cols; j++){
      data[i][j] = rand()%3-1;
    }
  }
}
void transpose_vector(vector<vector<int64_t>>& in, vector<vector<int64_t>>& out, int rows, int cols){
  for(int i = 0; i < cols; i++){
    vector<int64_t> row_data(rows);
    for(int j = 0; j < rows; j++){
      row_data[j] = in[j][i];
    }
    out[i] = row_data;
  }
}
void verify(vector<vector<int64_t>>& a, vector<vector<int64_t>>& b, int H, int W){
  for(int h = 0; h < H; h++){
    for(int w = 0; w < W; w++){
      if(a[h][w] != b[w][h]){
        cout << "verify failed: (" << h << "," << w << "), (" << a[h][w] << "," << b[w][h] << ")" << endl;
        assert(false);
      }
    }
  }
  cout << "verify success" << endl;
}
void verify_decrypted(vector<vector<int64_t>>& a, vector<Ciphertext>& b, int H, int W, MNIST_HE& mnist){
  for(int w = 0; w < W; w++){
    vector<int64_t> batchs(H);
    mnist.decrypted(b[w], batchs);
    for(int h = 0; h < H; h++){
      if(a[h][w] != batchs[h]){
        cout << "verify failed: (" << h << "," << w << "), (" << a[h][w] << "," << batchs[h] << ")" << endl;
        assert(false);
      }
    }
  }
  cout << "verify success" << endl;
}

template<typename T>
void print_vector(vector<vector<T>>& vec){
  for(int i = 0; i < vec.size(); i++){
    for(int j = 0; j < vec[i].size(); j++){
      cout << vec[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}
void test_gemm_batch(){
  int batchs = 1;
  int M = 100;
  int K = 1250;
  int N = 1;
  vector<vector<int64_t>> A(batchs), B(batchs), C(batchs);
  rand_input(A, batchs, M*K);
  rand_input(B, batchs, N*K);
  for(int b = 0; b < batchs; b++){
    C[b].resize(M*N);
  }
  vector<vector<int64_t>> trans_a(M*K), trans_b(K*N), trans_c(M*N);
  transpose_vector(A, trans_a, batchs, M*K);
  transpose_vector(B, trans_b, batchs, N*K);

  MNIST_HE mnist_he(8192, true);
  MNIST mnist;
  vector<Ciphertext> cip_b(K*N), cip_c(M*N);
  mnist_he.encrypted(trans_b, cip_b);
  mnist_he.fully_connected(trans_a, cip_b, cip_c, batchs, M, K);
  mnist.fully_connected(A, B, C, batchs, M, K);
  verify_decrypted(C, cip_c, batchs, M*N, mnist_he);
}
void test_mnist(){
  //init input data
  int batchs = 8192;
  int H = 28, W = 28;
  int KH = 5, KW = 5;
  int stride = 2;
  int pad_h = 1, pad_w = 1;
  int out_channels = 5;
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  vector<vector<int64_t>> images(batchs), kernels(batchs), outs(batchs);//, out2(batchs);
  rand_input(images, batchs, H*W);
  rand_weight(kernels, batchs, out_channels * KH * KW);

  for(int i = 0; i < batchs; i++){
    outs[i].resize(out_channels * OH * OW);
  }

  //
  MNIST_HE mnist_he(16384, true);
  MNIST mnist;

  mnist_he.test();
  //encrypted input image and weight
  vector<Ciphertext> image_encrypted(H*W), kernel_encrypted(out_channels * KH*KW), out_encrypted(out_channels * OH * OW);
  double start = omp_get_wtime();
  vector<vector<int64_t>> batch_images(H*W), batch_kernels(out_channels * KH*KW), batch_outs(out_channels*OH*OW);
  transpose_vector(images, batch_images, batchs, H*W);
  transpose_vector(kernels, batch_kernels, batchs, out_channels * KH * KW);
  for(int i = 0; i < out_channels*OH*OW; i++){
    batch_outs[i].resize(batchs);
  }
  mnist_he.encrypted(batch_images, image_encrypted);
  double end = omp_get_wtime();
  cout << "the times of encrypted image and kernel: " << (double)(end-start) << "s" << endl;

  start = omp_get_wtime();
  //infer
  //conv
  {
    mnist_he.conv_batch(image_encrypted, batch_kernels, out_encrypted, batchs, 1, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
    mnist.conv_batch(images, kernels, outs, batchs, 1, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
    verify_decrypted(outs, out_encrypted, batchs, out_channels*OH*OW, mnist_he);
    //print_vector(outs);
  }

  //activation
  {
    mnist_he.square_activation(out_encrypted);
    mnist.square_activation(outs);
    verify_decrypted(outs, out_encrypted, batchs, out_channels*OH*OW, mnist_he);
  }

  //pool
  int kernel_size = 3, padding = 1;
  H = OH; W = OW;
  OH = H + 2*padding - kernel_size + 1; 
  OW = W + 2*padding - kernel_size + 1;
  vector<vector<int64_t>> tmp_pool_out(batchs);
  vector<Ciphertext> pool_out(out_channels * OH *OW);
  {
    mnist_he.scaled_mean_pool(out_encrypted, pool_out, batchs, H, W, kernel_size, kernel_size, padding, out_channels);

    for(int i = 0; i < batchs; i++) tmp_pool_out[i].resize(out_channels * OH *OW);
    mnist.scaled_mean_pool(outs, tmp_pool_out, batchs, H, W, kernel_size, kernel_size, padding, out_channels);
    verify_decrypted(tmp_pool_out, pool_out, batchs, out_channels * OH * OW, mnist_he);
    //print_vector(tmp_pool_out);
  }

  //conv
  int IC = out_channels;
  out_channels = 10;
  H = OH; W = OW; pad_h = 0; pad_w = 0; stride = 2;
  OH = (H + 2*pad_h - KH) / stride + 1;
  OW = (W + 2*pad_w - KW) / stride + 1;
  vector<Ciphertext> conv_out2(IC*out_channels * OH * OW);
  {
    vector<vector<int64_t>> kernels2(batchs), batch_kernels2(out_channels*KH*KW);
    rand_weight(kernels2, batchs, out_channels*KH*KW);
    transpose_vector(kernels2, batch_kernels2, batchs, out_channels*KH*KW);
    mnist_he.conv_batch(pool_out, batch_kernels2, conv_out2, batchs, IC, H, W, KH, KW, stride, pad_h, pad_w, out_channels);

    for(auto& item : outs) item.resize(IC*out_channels * OH * OW);
    mnist.conv_batch(tmp_pool_out, kernels2, outs, batchs, IC, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
    verify_decrypted(outs, conv_out2, batchs, IC*out_channels*OH*OW, mnist_he);
    //print_vector(outs);
  }
  
  //pool
  vector<Ciphertext> pool_out2(IC*out_channels * OH * OW);
  {
    mnist_he.scaled_mean_pool(conv_out2, pool_out2, batchs, OH, OW, kernel_size, kernel_size, padding, IC*out_channels);

    for(auto& item : tmp_pool_out) item.resize(IC*out_channels * OH * OW); 
    mnist.scaled_mean_pool(outs, tmp_pool_out, batchs, OH, OW, kernel_size, kernel_size, padding, IC*out_channels);
    verify_decrypted(tmp_pool_out, pool_out2, batchs, IC*out_channels*OH*OW, mnist_he);
    //print_vector(tmp_pool_out);
  }

  {
    //mul
    //int out_nodes = 10;
    //vector<Ciphertext> fc_out(out_nodes), enc_weight(out_nodes*pool_out2.size());
    //vector<vector<int64_t>> fc_weight(batchs), batch_fc_weight(out_nodes*pool_out2.size()); 
    //rand_weight(fc_weight, batchs, out_nodes*pool_out2.size());
    //transpose_vector(fc_weight, batch_fc_weight, batchs, out_nodes*pool_out2.size());
    //mnist_he.mul(batch_fc_weight, pool_out2, enc_weight, batchs, out_nodes, pool_out2.size());
    //for(auto &item : outs) item.resize(out_nodes);
    //mnist.mul(fc_weight, tmp_pool_out, outs, batchs, out_nodes, pool_out2.size());
    //verify_decrypted(fc_weight, enc_weight, batchs, fc_weight[0].size(), mnist_he);

    //mnist_he.add(enc_weight, fc_out, batchs, out_nodes, pool_out2.size());
    //mnist.add(fc_weight, tmp_pool_out, outs, batchs, out_nodes, pool_out2.size());
    //verify_decrypted(outs, fc_out, batchs, fc_out.size(), mnist_he);
    //return;
  }
  //fully connected
  int out_nodes = 100;
  vector<Ciphertext> fc_out(out_nodes);
  {
    vector<vector<int64_t>> fc_weight(batchs), batch_fc_weight(out_nodes*pool_out2.size()); 
    rand_weight(fc_weight, batchs, out_nodes*pool_out2.size());
    transpose_vector(fc_weight, batch_fc_weight, batchs, out_nodes*pool_out2.size());
    mnist_he.fully_connected(batch_fc_weight, pool_out2, fc_out, batchs, out_nodes, pool_out2.size());

    for(auto &item : outs) item.resize(out_nodes);
    mnist.fully_connected(fc_weight, tmp_pool_out, outs, batchs, out_nodes, pool_out2.size());
    verify_decrypted(outs, fc_out, batchs, out_nodes, mnist_he);
    //print_vector(outs);
  }

  //activation
  {
    mnist_he.square_activation(fc_out);
    mnist.square_activation(outs);
    verify_decrypted(outs, fc_out, batchs, out_nodes, mnist_he);
  }

  //fully connected
  out_nodes = 10;
  vector<Ciphertext> fc_out2(out_nodes);
  {
    vector<vector<int64_t>> fc_weight2(batchs), batch_fc_weight2(out_nodes*fc_out.size()); 
    rand_weight(fc_weight2, batchs, out_nodes*fc_out.size());
    transpose_vector(fc_weight2, batch_fc_weight2, batchs, out_nodes*fc_out.size());
    mnist_he.fully_connected(batch_fc_weight2, fc_out, fc_out2, batchs, out_nodes, fc_out.size());

    for(auto& item : tmp_pool_out) item.resize(out_nodes);
    mnist.fully_connected(fc_weight2, outs, tmp_pool_out, batchs, out_nodes, fc_out.size());
    verify_decrypted(tmp_pool_out, fc_out2, batchs, out_nodes, mnist_he);
  }

  end = omp_get_wtime();
  cout << "infer time = " << end-start << "s" << endl;

  start = omp_get_wtime();
  vector<vector<int64_t>> decrypted_out(out_nodes);
  for(auto& item : decrypted_out) item.resize(batchs);
  mnist_he.decrypted(fc_out2, decrypted_out);
  end = omp_get_wtime();
  for(int i = 0; i < out_nodes; i++){
    cout << decrypted_out[i][0] << " ";
  }
  cout << endl;
  cout << "decrypted result time = " << end-start << "s" << endl;
}

int main(){
  test_mnist();
  return 0;
}
