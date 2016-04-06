/// @file       example.c
/// @author     Brian Cairl
/// @copyright  2015-2016
/// @brief      C-style example of NARXNet usage

#include <math.h>
#include <NARXNet/NARXNet.h>



int main(int argc, char** argv)
{
  NARXConfig my_narx_conf;

  // Configure the network
  conf.feature_len        = 3;   // number of coordinates in the process signal
  conf.input_len          = 2;   // number of coordinates in the input signal
  conf.order              = 3;   // number of input-layer delays
  conf.n_layers           = 2;   // number of layers (either 1 or 2, for now)
  conf.hidden_len_1       = 20;  // number of hidden neurons on first hidden layer
  conf.hidden_len_2       = 20;  // number of hidden neurons on second hidden layer
  conf.weight_init        = 0.1; // initial value of connection weightign coeffs
  conf.learning_rate_init = 0.001; // initial learning rate
  conf.momentum_init      = 0.001; // initial learning momentum

  NARXNet_t* my_narx_network;

  // Creates a new network
  NARXNet_Create(my_narx_network, &my_narx_conf);

  // Train
  float dt = 0.001;
  float t  = 0;
  size_t n = 1000;

  // Loop for n iterations
  while (n--)
  {
    // Create an input/proc signal 
    // (arbitrary; just for example's sake)
    
    // You might need to scale/shift real inputs/outputs
    float_t input_sig[2];
    input_sig[0] = cos(t);
    input_sig[1] = sin(t);

    // current process signal
    float_t proc_sig[3];
    proc_sig[0] = cos(t)*cos(t);
    proc_sig[1] = sin(t)*cos(t);
    proc_sig[2] = cos(t)*cos(t);

    // train
    NARXNet_Update(
      my_narx_network,
      proc_sig,
      input_sig,
      NARX_PARALLEL, // NARX_PARALLEL or NARX_SERIES_PARALLEL
      NARX_LR_ADAPTIVE_PROP | NARX_TRAIN // use adaption learning rate and train
    );

    // increment time
    t += dt;
  }

  const float_t t_r = 0.2;

  // You might need to scale/shift real inputs/outputs
  float_t input_sig_r[2];
  input_sig_r[0] = cos(t_r);
  input_sig_r[1] = sin(t_r);

  // current process signal
  float_t proc_sig_r[3];
  proc_sig_r[0] = cos(t_r)*cos(t_r);
  proc_sig_r[1] = sin(t_r)*cos(t_r);
  proc_sig_r[2] = cos(t_r)*cos(t_r);

  // To make ap prediction
  NARXNet_Update(
    my_narx_network,
    proc_sig_r,
    input_sig_r,
    NARX_PARALLEL, // NARX_PARALLEL or NARX_SERIES_PARALLEL
    NARX_PREDICT // use adaption learning rate and train
  );

  // Get process signal prediction (a 2-element array)
  const float_ptr pred_buffer = NARXNet_GetPrediction(my_narx_network);

  return 0;
}