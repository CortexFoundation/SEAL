#include "examples.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace seal;

class Test{
  private:
    shared_ptr<SEALContext> context;
    KeyGenerator *keygen;
    Evaluator *evaluator;
    PublicKey public_key;
    SecretKey secret_key;
    RelinKeys relin_keys;
    Encryptor *encryptor;
    Decryptor *decryptor;
    BatchEncoder *batch_encoder;
  public:
    Test(const int degree, bool batch = false){
      EncryptionParameters parms(scheme_type::BFV);
      if(!batch){
        size_t poly_modulus_degree = degree;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(1024);
      }else{
        size_t poly_modulus_degree = degree;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
      }
      context = SEALContext::Create(parms);
      auto qualifiers = context->first_context_data()->qualifiers();
      cout << "Batching enabled: " << boolalpha << qualifiers.using_batching << endl;

      evaluator = new Evaluator(context);
      keygen = new KeyGenerator(context);
      public_key = keygen->public_key();
      secret_key = keygen->secret_key();
      relin_keys = keygen->relin_keys_local();
      encryptor = new Encryptor(context, public_key);
      decryptor = new Decryptor(context, secret_key);
      batch_encoder = new BatchEncoder(context);
    }
    template<typename T>
    void encrypted(vector<T>& data, vector<Ciphertext>& ecp_data){
      cout << "encrypted data size = " << data.size() << endl;
      for(int i = 0; i < data.size(); i++){
        int x = data[i];
        Plaintext x_plain(to_string(x));
        Ciphertext x_encrypted;
        encryptor->encrypt(x_plain, x_encrypted);
        ecp_data[i] = x_encrypted;
      }
    }
    template<typename T>
    void encrypted(vector<vector<T>>& data, vector<Ciphertext>& ecp_data){
      cout << "encrypted data size = " << data.size() << endl;
#pragma omp parallel for
      for(int i = 0; i < data.size(); i++){
        Ciphertext vec_encrypted;
        Plaintext vec_plain;
        batch_encoder->encode(data[i], vec_plain);
        encryptor->encrypt(vec_plain, vec_encrypted);
        ecp_data[i] = vec_encrypted;
      }
    }
    int hex_to_int(const string& hex){
      int value = 0;
      int x = 1;
      for(int i = hex.size()-1; i>=0; i--){
        int c = 0;
        if(hex[i] >= '0' && hex[i] <= '9')
          c = hex[i] - '0';
        else if(hex[i] >= 'A' && hex[i] <= 'F')
          c = hex[i] - 'A' + 10;
        else c = hex[i] - 'a' + 10;
        value += x * c; 
        x *= 16;
      }
      return value;
    }
    template<typename T>
    void decrypted(vector<Ciphertext>& ecp_data, vector<T>& data){
      for(int i = 0; i < ecp_data.size(); i++){
        Plaintext x_decrypted;
        decryptor->decrypt(ecp_data[i], x_decrypted);
        data[i] = hex_to_int(x_decrypted.to_string());
      }
    }
    template<typename T>
    void decrypted(Ciphertext& ecp_data, vector<T>& data){
      Plaintext vec_decrypted;
      decryptor->decrypt(ecp_data, vec_decrypted);
      batch_encoder->decode(vec_decrypted, data);
    }
    template<typename T>
    void decrypted(vector<Ciphertext>& ecp_data, vector<vector<T>>& data){
      for(int i = 0; i < ecp_data.size(); i++){
        Plaintext vec_decrypted;
        decryptor->decrypt(ecp_data[i], vec_decrypted);
        //data[i] = hex_to_int(x_decrypted.to_string());
        batch_encoder->decode(vec_decrypted, data[i]);
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
    void gemm(vector<int>& A, vector<int>& B, vector<int>& C, const int M, const int K, const int N){
      cout << "test gemm..." << endl;
      cout << "M=" << M << " K=" << K << " N=" << N << endl;
      vector<Ciphertext> a_encrypted(A.size()), b_encrypted(B.size()), c_encrypted(C.size());
      clock_t start = clock();
      encrypted(A, a_encrypted);
      encrypted(B, b_encrypted);
      clock_t end = clock();
      cout << "the times of encrypted matrix A and B: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;

      start = clock();
      for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
          Plaintext zero("0");
          Ciphertext sum;
          encryptor->encrypt(zero, sum); 
          for(int k = 0; k < K; k++){
            Ciphertext axb;
            evaluator->multiply(a_encrypted[i*K+k], b_encrypted[k*N+j], axb);
            evaluator->add_inplace(sum, axb);
          } 
          c_encrypted[i*N+j] = sum;
        }
      }
      end = clock();
      cout << "the times of calculation: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;
      start = clock();
      decrypted(c_encrypted, C);
      end = clock();
      cout << "the times of descrypted matrix C: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;
    }
    void gemm_batch(vector<vector<int64_t>>& A, vector<vector<int64_t>>& B, vector<vector<int64_t>>& C, const int batchs, const int M, const int K, const int N){
      cout << "test gemm..." << endl;
      cout << "M=" << M << " K=" << K << " N=" << N << endl;
      vector<vector<int64_t>> batch_a(M*K), batch_b(K*N), batch_c(M*N);
      cout << "transpose ..." << endl;
      transpose_vector(A, batch_a, batchs, M*K);
      transpose_vector(B, batch_b, batchs, K*N);
      
      vector<Ciphertext> a_encrypted(batch_a.size()), b_encrypted(batch_b.size()), c_encrypted(batch_c.size());
      double start = omp_get_wtime();
     // encrypted(batch_a, a_encrypted);
      encrypted(batch_b, b_encrypted);
      double end = omp_get_wtime();
      cout << "the times of encrypted matrix A and B: " << (double)(end-start) << "s" << endl;

      start = omp_get_wtime();
#pragma omp parallel for collapse(2)
      for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
          Plaintext zero("0");
          Ciphertext sum;
          encryptor->encrypt(zero, sum); 
          for(int k = 0; k < K; k++){
            Ciphertext axb;
            Plaintext plain_a;
            batch_encoder->encode(batch_a[i*K+k], plain_a);
            evaluator->multiply_plain(b_encrypted[k*N+j], plain_a, axb);
            evaluator->add_inplace(sum, axb);
            evaluator->relinearize_inplace(sum, relin_keys);

      ///      vector<int64_t> vec_sum(batchs); 
      ///      vector<int64_t> vec_a(batchs); 
      ///      vector<int64_t> vec_b(batchs); 
      ///      vector<int64_t> vec_axb(batchs); 
      ///      decrypted(a_encrypted[i*K+k], vec_a);
      ///      decrypted(b_encrypted[k*N+j], vec_b);
      ///      decrypted(axb, vec_axb);
      ///      decrypted(sum, vec_sum);
      ///      printf("%d %d %d, %lld * %lld = %lld, %lld\n", i, j, k, vec_a[0], vec_b[0], vec_axb[0], vec_sum[0]);
          } 
          c_encrypted[i*N+j] = sum;
        }
      }
      end = omp_get_wtime();
      cout << "the times of calculation: " << (double)(end-start) << "s" << endl;
      start = omp_get_wtime();
      for(auto& item : batch_c) item.resize(batchs);
      decrypted(c_encrypted, batch_c);
      transpose_vector(batch_c, C, M*N, batchs);
      end = omp_get_wtime();
      cout << "the times of descrypted matrix C: " << (double)(end-start) << "s" << endl;
    }
    void conv(vector<int>& image, vector<int>& kernel, vector<int>& out, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
      cout << "test conv..." << endl;
      cout << "image(" << H << "," << W << "), kernel(" << KH << "," << KW << "), stride=" << stride << ", out channels = " << out_channels << endl;
      vector<Ciphertext> image_encrypted(image.size()), kernel_encrypted(kernel.size()), out_encrypted(out.size());
      clock_t start = clock();
      encrypted(image, image_encrypted);
      encrypted(kernel, kernel_encrypted);
      clock_t end = clock();
      cout << "the times of encrypted image and kernel: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;

      start = clock();
      int OH = (H + 2*pad_h - KH) / stride + 1;
      int OW = (W + 2*pad_w - KW) / stride + 1;
      for(int oc = 0; oc < out_channels; oc++){
        for(int oh = 0; oh < OH; oh++){
          for(int ow = 0; ow < OW; ow++){
            //int sum = 0;
            Plaintext zero("0");
            Ciphertext sum;
            encryptor->encrypt(zero, sum); 
            for(int kh = 0; kh < KH; kh++){
              for(int kw = 0; kw < KW; kw++){
                int ih = oh * stride + kh - pad_h;
                int iw = ow * stride + kw - pad_w;
                if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                }else{
                  //sum += image[ih*W + iw] * kernel[oc*KH*KW + kh*KW + kw];  
                  Ciphertext axb;
                  evaluator->multiply(image_encrypted[ih*W+iw], kernel_encrypted[oc*KH*KW + kh*KW + kw], axb);
                  evaluator->add_inplace(sum, axb);
                } 
              }
            }
            //out[oc*OH*OW + oh*OW + ow] = sum;
            out_encrypted[oc*OH*OW + oh*OW + ow] = sum;
          }
        }
      } 
      end = clock();
      cout << "the times of calculation: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;
      start = clock();
      decrypted(out_encrypted, out);
      end = clock();
      cout << "the times of encrypted matrix C: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;
    }
    void conv_batch(vector<vector<uint64_t>>& images, vector<vector<uint64_t>>& kernels, vector<vector<uint64_t>>& outs, int batchs, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
      int OH = (H + 2*pad_h - KH) / stride + 1;
      int OW = (W + 2*pad_w - KW) / stride + 1;
      cout << "test conv..." << endl;
      cout << "batchs=" << batchs << ", image(" << H << "," << W << "), kernel(" << KH << "," << KW << "), stride=" << stride << ", out channels = " << out_channels << endl;
      vector<Ciphertext> image_encrypted(H*W), kernel_encrypted(out_channels * KH*KW), out_encrypted(out_channels * OH * OW);
      clock_t start = clock();
      vector<vector<uint64_t>> batch_images(H*W), batch_kernels(out_channels * KH*KW), batch_outs(out_channels*OH*OW);
      for(int i = 0; i < H*W; i++){
        vector<uint64_t> batch_pixels(batchs);
        for(int j = 0; j < batchs; j++){
          batch_pixels[j] = images[j][i];
        }
        batch_images[i] = batch_pixels;
      }
      for(int i = 0; i < out_channels * KH*KW; i++){
        vector<uint64_t> batch_pixels(batchs);
        for(int j = 0; j < batchs; j++){
          batch_pixels[j] = kernels[j][i];
        }
        batch_kernels[i] = batch_pixels;
      }
      for(int i = 0; i < out_channels*OH*OW; i++){
        batch_outs[i].resize(batchs);
      }
      encrypted(batch_images, image_encrypted);
      encrypted(batch_kernels, kernel_encrypted);
      clock_t end = clock();
      cout << "the times of encrypted image and kernel: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;

      vector<uint64_t> zero(batchs, 0);
      Plaintext zero_plaintext;
      batch_encoder->encode(zero, zero_plaintext);
      Ciphertext zero_enc;
      encryptor->encrypt(zero_plaintext, zero_enc);
      //start = clock();
      double omp_start = omp_get_wtime();
#pragma omp parallel for collapse(3) shared(image_encrypted, kernel_encrypted, zero_enc, out_encrypted)
      for(int oc = 0; oc < out_channels; oc++){
        for(int oh = 0; oh < OH; oh++){
          for(int ow = 0; ow < OW; ow++){
            Ciphertext sum = zero_enc;
            for(int kh = 0; kh < KH; kh++){
              for(int kw = 0; kw < KW; kw++){
                int ih = oh * stride + kh - pad_h;
                int iw = ow * stride + kw - pad_w;
                if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                }else{
                  Ciphertext axb;
                  evaluator->multiply(image_encrypted[ih*W+iw], kernel_encrypted[oc*KH*KW + kh*KW + kw], axb);
                  evaluator->add_inplace(sum, axb);
                } 
              }
              evaluator->relinearize_inplace(sum, relin_keys);
            }
            out_encrypted[oc*OH*OW + oh*OW + ow] = sum;
          }
        }
      } 
      //end = clock();
      double omp_end = omp_get_wtime();
      cout << "the times of calculation: " << (double)(omp_end-omp_start) << "s" << endl;
      start = clock();
      
      decrypted(out_encrypted, batch_outs);
      for(int i = 0; i < out_channels * OH * OW; i++){
        for(int j = 0; j < batchs; j++){
          outs[j][i] = batch_outs[i][j];
        }
      }
      end = clock();
      cout << "the times of decrypted out: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << endl;
    }
    void verify(vector<int>& a, vector<int>& b){
      bool flag = true;
      if(a.size() != b.size()) flag = false;

      for(int i = 0; i < a.size(); i++){
        if(a[i] != b[i]){
          flag = false;
          break;
        }
      }
      if(flag){
        cout << "success" << endl;
      }else cout << "failed" << endl;
    }
    template<typename T>
    void verify(vector<vector<T>>& a, vector<vector<T>>& b){
      bool flag = true;
      if(a.size() != b.size()) {
        flag = false;
        cout << "vector size no equal" << endl;
      }

      for(int i = 0; i < a.size(); i++){
        if(a[i].size() != b[i].size()){
          cout << "vector size no equal" << endl;
          flag = false; break;
        }
        for(int j = 0; j < a[i].size(); j++){
          if(a[i][j] != b[i][j]){
            flag = false;
            cout << i << "," << j << ": " << a[i][j] << ", " << b[i][j] << endl;
            break;
          }
        }
        if(!flag) break;
      }
      if(flag){
        cout << "success" << endl;
      }else cout << "failed" << endl;
    }
};

void gemm(vector<int>& A, vector<int>& B, vector<int>& C, const int M, const int K, const int N){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < K; j++){
      int sum = 0;
      for(int k = 0; k < N; k++){
        sum += A[i*K+k] * B[k*N+j];
      }
      C[i*N+j] = sum;
    }
  }
}
void gemm_batch(vector<vector<int64_t>>& A, vector<vector<int64_t>>& B, vector<vector<int64_t>>& C, const int batchs, const int M, const int K, const int N){
  for(int b = 0; b < batchs; b++){
    for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
        int64_t sum = 0;
        for(int k = 0; k < K; k++){
          int64_t tmp = A[b][i*K+k] * B[b][k*N+j];
          sum += tmp;
          //printf("%d %d %d, %lld * %lld = %lld, %lld\n", i, j, k, A[b][i*K+k], B[b][k*N+j], tmp, sum);
        }
        C[b][i*N+j] = sum;
      }
    }
  }
}
void test_gemm(){
  int M = 8;
  int K = 8;
  int N = 8;
  vector<int> A(M*K), B(K*N), C(M*N);
  for(int i = 0; i < M*K; i++){
    A[i] = rand() % 10;
  }
  for(int i = 0; i < K*N; i++){
    B[i] = rand() % 10;
  }

  Test test(4096);
  test.gemm(A, B, C, M, K, N);
  vector<int> C1(M*N);
  gemm(A, B, C1, M, K, N);
  test.verify(C, C1);
}
void test_gemm_batch(){
  int batchs = 8192;
  int M = 100;
  int K = 1250;
  int N = 1;
  vector<vector<int64_t>> A(batchs), B(batchs), C(batchs), C1(batchs);
  for(int b = 0; b < batchs; b++){
    A[b].resize(M*K);
    for(int i = 0; i < M*K; i++){
      A[b][i] = rand() % 10;
    }
    B[b].resize(K*N);
    for(int i = 0; i < K*N; i++){
      B[b][i] = rand() % 10;
    }
    C[b].resize(M*N);
    C1[b].resize(M*N);
  }

  Test test(8192, true);
  test.gemm_batch(A, B, C, batchs, M, K, N);
  gemm_batch(A, B, C1, batchs, M, K, N);
  test.verify(C, C1);
}
void conv(vector<int>& image, vector<int>& kernel, vector<int>& out, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  for(int oc = 0; oc < out_channels; oc++){
    for(int oh = 0; oh < OH; oh++){
      for(int ow = 0; ow < OW; ow++){
        int sum = 0;
        for(int kh = 0; kh < KH; kh++){
          for(int kw = 0; kw < KW; kw++){
            int ih = oh * stride + kh - pad_h;
            int iw = ow * stride + kw - pad_w;
            if(ih < 0 || ih >= H || iw < 0 || iw >= W){
            }else{
              sum += image[ih*W + iw] * kernel[oc*KH*KW + kh*KW + kw];  
            } 
          }
        }
        out[oc*OH*OW + oh*OW + ow] = sum;
      }
    }
  } 
}

void conv_batch(vector<vector<uint64_t>>& images, vector<vector<uint64_t>>& kernels, vector<vector<uint64_t>>& outs, int batchs, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  for(int b = 0; b < batchs; b++){
    for(int oc = 0; oc < out_channels; oc++){
      for(int oh = 0; oh < OH; oh++){
        for(int ow = 0; ow < OW; ow++){
          uint64_t sum = 0;
          for(int kh = 0; kh < KH; kh++){
            for(int kw = 0; kw < KW; kw++){
              int ih = oh * stride + kh - pad_h;
              int iw = ow * stride + kw - pad_w;
              if(ih < 0 || ih >= H || iw < 0 || iw >= W){
              }else{
                sum += images[b][ih*W + iw] * kernels[b][oc*KH*KW + kh*KW + kw];  
              } 
            }
          }
          //outs[b*out_channels*OH*OW + oc*OH*OW + oh*OW + ow] = sum;
          outs[b][oc*OH*OW + oh*OW + ow] = sum;
        }
      }
    } 
  }
}
void test_conv(){
  int H = 28, W = 28;
  int KH = 5, KW = 5;
  int stride = 2;
  int pad_h = 1, pad_w = 1;
  int out_channels = 5;
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  vector<int> image(H*W), kernel(out_channels * KH*KW), out(out_channels*OH*OW);
  conv(image, kernel, out, H, W, KH, KW, stride, pad_h, pad_w, out_channels);

  vector<int> out2(out.size());
  
  Test test(2048);
  test.conv(image, kernel, out2, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
  test.verify(out, out2);
}
void test_conv_batch(){
  int batchs = 8192;
  int H = 28, W = 28;
  int KH = 5, KW = 5;
  int stride = 2;
  int pad_h = 1, pad_w = 1;
  int out_channels = 5;
  int OH = (H + 2*pad_h - KH) / stride + 1;
  int OW = (W + 2*pad_w - KW) / stride + 1;
  vector<vector<uint64_t>> image(batchs), kernel(batchs), out(batchs), out2(batchs);
  for(int i = 0; i < batchs; i++){
    image[i].resize(H*W);
    for(int j = 0; j < image[i].size(); j++){
      image[i][j] = rand() % 10; 
    }
    kernel[i].resize(out_channels * KH *KW);
    for(int j = 0; j < kernel[i].size(); j++){
      kernel[i][j] = rand() % 10; 
    }
    out[i].resize(out_channels * OH * OW);
    out2[i].resize(out_channels * OH * OW);
  }
  conv_batch(image, kernel, out, batchs, H, W, KH, KW, stride, pad_h, pad_w, out_channels);

  Test test(8192, true);
  test.conv_batch(image, kernel, out2, batchs, H, W, KH, KW, stride, pad_h, pad_w, out_channels);
  test.verify(out, out2);
}
int main(){
  //test_gemm();
  test_gemm_batch();
  //test_conv_batch();
  return 0;
}
