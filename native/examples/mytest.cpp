#include "examples.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace seal;

class Test{
  private:
    shared_ptr<SEALContext> context;
    KeyGenerator *keygen;
    Evaluator *evaluator;
    PublicKey public_key;
    SecretKey secret_key;
    Encryptor *encryptor;
    Decryptor *decryptor;
  public:
    Test(const int degree){
      EncryptionParameters parms(scheme_type::BFV);
      size_t poly_modulus_degree = degree;
      parms.set_poly_modulus_degree(poly_modulus_degree);
      parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
      parms.set_plain_modulus(1024);
      context = SEALContext::Create(parms);
      evaluator = new Evaluator(context);
      keygen = new KeyGenerator(context);
      public_key = keygen->public_key();
      secret_key = keygen->secret_key();
      encryptor = new Encryptor(context, public_key);
      decryptor = new Decryptor(context, secret_key);
    }
    void encrypted(vector<int>& data, vector<Ciphertext>& ecp_data){
      for(int i = 0; i < data.size(); i++){
        int x = data[i];
        Plaintext x_plain(to_string(x));
        Ciphertext x_encrypted;
        encryptor->encrypt(x_plain, x_encrypted);
        ecp_data[i] = x_encrypted;
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
    void decrypted(vector<Ciphertext>& ecp_data, vector<int>& data){
      for(int i = 0; i < ecp_data.size(); i++){
        Plaintext x_decrypted;
        decryptor->decrypt(ecp_data[i], x_decrypted);
        data[i] = hex_to_int(x_decrypted.to_string());
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
int main(){
  test_gemm();
  test_conv();
  return 0;
}
