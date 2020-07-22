#ifndef MNIST_HE_H
#define MNIST_HE_H

#include "examples.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace seal;

class MNIST_HE{
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
    int64_t module_value;
  public:
    MNIST_HE(const int degree, bool batch = false){
      EncryptionParameters parms(scheme_type::BFV);
      size_t poly_modulus_degree = degree;
      parms.set_poly_modulus_degree(poly_modulus_degree);
      vector<Modulus> coeff_modul = CoeffModulus::BFVDefault(poly_modulus_degree);
//      for(int i = 0; i < coeff_modul.size(); i++)
//      cout << coeff_modul[i].value() << endl;
      parms.set_coeff_modulus(coeff_modul);
      //parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
      if(!batch){
        parms.set_plain_modulus(1024);
      }else{
        //parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        Modulus module = PlainModulus::Batching(poly_modulus_degree, 42);
        //parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        parms.set_plain_modulus(module);
        module_value = module.value();
//        cout << module_value << endl;
      }

      context = SEALContext::Create(parms);
      auto qualifiers = context->first_context_data()->qualifiers();
      cout << "Batching enabled: " << boolalpha << qualifiers.using_batching << endl;
      print_parameters(context);

      evaluator = new Evaluator(context);
      keygen = new KeyGenerator(context);
      public_key = keygen->public_key();
      secret_key = keygen->secret_key();
      relin_keys = keygen->relin_keys_local();
      encryptor = new Encryptor(context, public_key);
      decryptor = new Decryptor(context, secret_key);
      batch_encoder = new BatchEncoder(context);
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
    void encrypted(vector<vector<int64_t>>& data, vector<Ciphertext>& ecp_data){
#pragma omp parallel for
      for(int i = 0; i < data.size(); i++){
        //Ciphertext vec_encrypted;
        Plaintext vec_plain;
        batch_encoder->encode(data[i], vec_plain);
        encryptor->encrypt(vec_plain, ecp_data[i]);
      //  ecp_data[i] = vec_encrypted;
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
    void decrypted(Ciphertext& ecp_data, vector<int64_t>& data){
      Plaintext vec_decrypted;
      decryptor->decrypt(ecp_data, vec_decrypted);
      batch_encoder->decode(vec_decrypted, data);
    }
    void decrypted(vector<Ciphertext>& ecp_data, vector<vector<int64_t>>& data){
      for(int i = 0; i < ecp_data.size(); i++){
        Plaintext vec_decrypted;
        decryptor->decrypt(ecp_data[i], vec_decrypted);
        batch_encoder->decode(vec_decrypted, data[i]);
      }
    }

    void conv_batch(vector<Ciphertext>& image_encrypted, vector<vector<int64_t>>& kernel, vector<Ciphertext>& out_encrypted, int batchs, int IC, int H, int W, int KH, int KW, int stride, int pad_h, int pad_w, int out_channels){
      int OH = (H + 2*pad_h - KH) / stride + 1;
      int OW = (W + 2*pad_w - KW) / stride + 1;
      cout << "Convolution Layer.. " << endl;
      cout << "batchs=" << batchs << ", image(" << H << "," << W << "), kernel(" << KH << "," << KW << "), stride=" << stride << ", out channels = " << out_channels << ", (OH, OW)=(" << OH << "," << OW << ")" << endl;
      vector<int64_t> zero(batchs, 0);
      Plaintext zero_plaintext;
      batch_encoder->encode(zero, zero_plaintext);
      Ciphertext zero_enc;
      encryptor->encrypt(zero_plaintext, zero_enc);
      //start = clock();
      double omp_start = omp_get_wtime();
#pragma omp parallel for collapse(3) shared(image_encrypted, kernel, zero_enc, out_encrypted)
      for(int oc = 0; oc < out_channels; oc++){
        for(int oh = 0; oh < OH; oh++){
          for(int ow = 0; ow < OW; ow++){
            for(int ic = 0; ic < IC; ic++){
              Ciphertext sum = zero_enc;
              for(int kh = 0; kh < KH; kh++){
                for(int kw = 0; kw < KW; kw++){
                  int ih = oh * stride + kh - pad_h;
                  int iw = ow * stride + kw - pad_w;
                  if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                  }else{
                    Ciphertext axb;
                    Plaintext plain_kernel;
                    batch_encoder->encode(kernel[oc*KH*KW+kh*KW+kw], plain_kernel);
                    evaluator->multiply_plain(image_encrypted[ic*H*W + ih*W+iw], plain_kernel, axb);
                    evaluator->relinearize_inplace(axb, relin_keys);
                    evaluator->add_inplace(sum, axb);
                  } 
                }
                evaluator->relinearize_inplace(sum, relin_keys);
              }
              out_encrypted[(oc*IC + ic)*OH*OW + oh*OW + ow] = sum;
            }
          }
        }
      } 
      //end = clock();
      double omp_end = omp_get_wtime();
      cout << "The times of convolution layer: " << (double)(omp_end-omp_start) << "s" << endl;
    }
    void square_activation(vector<Ciphertext>& data){
      cout << "Square Activatoin Layer..." << endl;
      double start = omp_get_wtime();
#pragma omp parallel for
      for(int i = 0; i < data.size(); i++){
        Ciphertext sq;
        evaluator->square(data[i], sq);
        //if(i < 5)
        //  cout << decryptor->invariant_noise_budget(data[i]) << endl;
        evaluator->relinearize_inplace(sq, relin_keys);
        data[i] = sq;
      }
      double end = omp_get_wtime();
      cout << "The times of square activation layer: " << end - start << "s" << endl;
    }
    void scaled_mean_pool(vector<Ciphertext>& input, vector<Ciphertext>& output,
        int batchs, int H, int W, int KH, int KW, int padding, int out_channels){
      cout << "Scaled Mean Pool Layer..." << endl;
      double start = omp_get_wtime();
      int OH = H + 2 * padding - KH + 1;
      int OW = W + 2 * padding - KW + 1;
      vector<int64_t> zero(batchs, 0);
      Plaintext zero_plaintext;
      batch_encoder->encode(zero, zero_plaintext);
      Ciphertext zero_enc;
      encryptor->encrypt(zero_plaintext, zero_enc);
#pragma omp parallel for collapse(3)
      for(int oc = 0; oc < out_channels; oc++){
        for(int oh = 0; oh < OH; oh++){
          for(int ow = 0; ow < OW; ow++){
            Ciphertext sum = zero_enc;
            for(int kh = 0; kh < KH; kh++){
              for(int kw = 0; kw < KW; kw++){
                int ih = oh + kh - padding;
                int iw = ow + kw - padding;
                if(ih < 0 || ih >= H || iw < 0 || iw >= W){
                }else{
                  evaluator->add_inplace(sum, input[oc*H*W+ih*W + iw]);
                }
              }
              evaluator->relinearize_inplace(sum, relin_keys);
            } 
            output[oc * OH*OW + oh*OW + ow] = sum;
          }
        } 
      }
      double end = omp_get_wtime();
      cout << "the times of scaled mean pool layer: " << end - start << "s" << endl;
    }
    void mul(vector<vector<int64_t>>& weight, vector<Ciphertext>& input, vector<Ciphertext>& output, int batchs, int H, int W){
      cout << H << " " << W << " " << weight.size() << " " << weight[0].size() << " " << input.size() <<  endl;
      cout << "module max value = " << module_value << endl;
//      FILE *fp = fopen("b.txt", "w");
//      FILE *fp2 = fopen("c.txt", "w");
      double start = omp_get_wtime();
//#pragma omp parallel for
      for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
          Ciphertext axb;
          Plaintext plain_w;
          vector<int64_t> one(batchs, 1);
          for(int k = 0; k < batchs; k++){
            one[k] = weight[i*W+j][k];
//            fprintf(fp2, "%ld ", one[k]);
          }
        
//          fprintf(fp2, "\n");
          batch_encoder->encode(one, plain_w);
          //batch_encoder->encode(weight[i*W+j], plain_w);
          evaluator->multiply_plain(input[j], plain_w, axb);
          //evaluator->add_plain(input[j], plain_w, axb);
          evaluator->relinearize_inplace(axb, relin_keys);
//          vector<int64_t> vec_input(batchs), vec_axb(batchs);
//          decrypted(input[j], vec_input);
//          decrypted(axb, vec_axb);
//          fprintf(fp, "%ld %ld %ld\n", one[0], vec_input[0], vec_axb[0]);
          output[i*W+j] = axb; 
        } 
      }
//      fclose(fp);
//      fclose(fp2);
      double end = omp_get_wtime();
      cout << "mul layer: " << end - start << "s" << endl;
    }
    void add(vector<Ciphertext>& input, vector<Ciphertext>& output, int batchs, int H, int W){
      cout << "module max value = " << module_value << endl;
      FILE *fp = fopen("b.txt", "w");
      double start = omp_get_wtime();
//#pragma omp parallel for
      for(int i = 0; i < H; i++){
        Plaintext zero("0");
        Ciphertext sum;
        encryptor->encrypt(zero, sum); 
        for(int j = 0; j < W; j++){
          if(i == 0 && j < 10)
          cout << decryptor->invariant_noise_budget(input[i*W+j]) << endl;

          evaluator->add_inplace(sum, input[i*W+j]);
//          vector<int64_t> vec_input(batchs), vec_sum(batchs);
//          decrypted(input[i*W+j], vec_input);
//          decrypted(sum, vec_sum);
//          fprintf(fp, "%ld %ld\n", vec_input[0], vec_sum[0]);
          evaluator->relinearize_inplace(sum, relin_keys);
        } 
        output[i] = sum;
      }

      fclose(fp);
      double end = omp_get_wtime();
      cout << "add layer: " << end - start << "s" << endl;
    }

    void fully_connected(vector<vector<int64_t>>& weight, vector<Ciphertext>& input, vector<Ciphertext>& output, int batchs, int H, int W){
      cout << "Fully Connected Layer..." << endl;
      cout << H << " " << W << " " << weight.size() << " " << weight[0].size() << " " << input.size() <<  endl;
      //cout << "module max value = " << module_value << endl;
      double start = omp_get_wtime();
#pragma omp parallel for
      for(int i = 0; i < H; i++){
        Plaintext zero("0");
        Ciphertext sum;
        encryptor->encrypt(zero, sum); 
        for(int j = 0; j < W; j++){
          Ciphertext axb;
          Plaintext plain_w;
          batch_encoder->encode(weight[i*W+j], plain_w);
          evaluator->multiply_plain(input[j], plain_w, axb);
          evaluator->relinearize_inplace(axb, relin_keys);
          //if(i == 0 && j < 5)
          //cout << decryptor->invariant_noise_budget(input[j]) << " " << decryptor->invariant_noise_budget(axb) << endl;
          evaluator->add_inplace(sum, axb);
          evaluator->relinearize_inplace(sum, relin_keys);
        } 
        output[i] = sum;
      }

      double end = omp_get_wtime();
      cout << "The times of fully connected layer: " << end - start << "s" << endl;
    }

    void test(){
      vector<int64_t> one(8192, 1), input(8192);
      for(int i = 0; i < 8192; i++){
        one[i] = rand() % 2 + 1;
        input[i] = rand() % 60000 + 60000;
      }
      Plaintext plain_one;
      batch_encoder->encode(one, plain_one);
      Plaintext plain_input;
      batch_encoder->encode(input, plain_input);
      Ciphertext input_enc;
      encryptor->encrypt(plain_input, input_enc);
      cout << decryptor->invariant_noise_budget(input_enc) << endl;
      Ciphertext y;
      evaluator->multiply_plain(input_enc, plain_one, y);
      vector<int64_t> vec_y(8192);
      decrypted(y, vec_y);
      cout << one[0] << " " << input[0] << " " << vec_y[0] << " " << vec_y[1] << endl;
    }
};

#endif
