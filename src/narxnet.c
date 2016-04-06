#include <math.h>
#include "narxnet/narxnet.h"
/// @todo Extend to a cascade network option
/// @todo Extend to a space network option



/// @section NARX_GLOBAL_x
/// @brief  Fixed, global definitions. Change at your own risk.
/// @{
#define NARX_GLOBAL_TRAINING_ALGORITHM          FANN_TRAIN_INCREMENTAL
#define NARX_GLOBAL_ERROR_FUNCTION              FANN_ERRORFUNC_TANH
#define NARX_GLOBAL_HIDDEN_ACTIVATION_FUNCTION  FANN_GAUSSIAN_SYMMETRIC
#define NARX_GLOBAL_OUTPUT_ACTIVATION_FUNCTION  FANN_LINEAR
#define NARX_BOLD_LAMBDA                        0.002f
#define NARX_PROP_LAMBDA                        0.1f
/// @}



/// @brief  Shifts entire blocks within a sequential buffer and inserts a new item 
///  @todo  Add to vector library (im not even sure which one is most recent at this point)
///  @param  buffer    a sequential buffer of inline blocks of length 'block_len'
/// @param  item    a new item to add to the buffer of length 'block_len'
/// @param  offset    offset of targeted input set
/// @param  block_len  the length of a single block
///  @param  n_blocks  number of blocks in the buffer
static void tapped_input_buffer_insert(float_ptr buffer, float_ptr item, const offset_t offset, const offset_t block_len, const offset_t n_blocks)
{

  unsigned int idx;

  /// Get swap stop-position
  unsigned int len_end = (n_blocks-1UL)*block_len;
  
  /// Start buffer at input offset
  buffer+=offset;

  /// Copy blocks back
  for(idx = 0UL; idx < len_end; idx++)
  {
    buffer[idx] = buffer[idx+block_len];  
  }

  /// Insert new item
  memcpy((buffer+len_end),item,block_len*sizeof(float_t));
}



static void scale_up(float_ptr buffer, float_ptr scale_value, offset_t len)
{
  while (len--)
  {
    *(buffer+len) *= (*scale_value);
  }
}


static void scale_down(float_ptr buffer, float_ptr scale_value, offset_t len)
{
  while (len--)
  {
    *(buffer+len) /= (*scale_value);
  }
}


static void diff(float_ptr dst, float_ptr src1, float_ptr src2, offset_t len)
{
  while (len--)
  {
    *(dst+len) = *(src2+len) - *(src1+len);
  }
}


#if NARX_OPT_COMPUTE_MSE
const float_t NARXNet_GetMSE(NARXNet_t* net)
{
  return net->MSE;
}
#endif


const float_ptr NARXNet_GetPrediction(NARXNet* net)
{
  return net->output_buffer;
}


void NARXNet_Create(
  NARXNet_t*    net, 
  NARXConfig_t*  conf
){
  /// Compute Necessary Input Buffer Length
  net->input_buffer_len  = ((conf->input_len+conf->sig_len) * conf->order);
  
  /// Input-buffer Positional Information
  net->network_order    = conf->order;
  net->input_block_len  = conf->input_len;
  net->output_block_len = conf->sig_len;
  net->input_offset     = (0UL);
  net->output_offset    = (conf->input_len*conf->order);

  /// Compute Necessary Output Buffer Length
  net->output_buffer_len= conf->sig_len;
  
  /// Allocate Input Buffer + copy swap
  net->input_buffer      = (float_ptr)calloc(net->input_buffer_len,  sizeof(float_t));

  /// Allocate Output Buffer + copy swap
  net->output_buffer      = (float_ptr)calloc(net->output_buffer_len,  sizeof(float_t));
  net->output_buffer_prev = (float_ptr)calloc(net->output_buffer_len,  sizeof(float_t));
  net->output_buffer_diff = (float_ptr)calloc(net->output_buffer_len,  sizeof(float_t));

  /// Create Base-FFNN
  if (conf->n_hidden_layers==2UL)
    net->network = fann_create_standard(2UL+conf->n_hidden_layers,net->input_buffer_len,conf->hidden_len_1,conf->hidden_len_2,net->output_buffer_len);
  else
    net->network = fann_create_standard(2UL+conf->n_hidden_layers,net->input_buffer_len,conf->hidden_len_1,net->output_buffer_len);
  
  /// Setup the Base-FFNN
  fann_randomize_weights(net->network,-conf->weight_init,+conf->weight_init);
  fann_set_learning_momentum(net->network, conf->momentum_init);
  fann_set_learning_rate(net->network, conf->learning_rate_init);

  /// Fixed Global Configs (@see NARX_GLOBAL_x) 
  fann_set_training_algorithm(net->network, NARX_GLOBAL_TRAINING_ALGORITHM);
  fann_set_activation_function_hidden(net->network, NARX_GLOBAL_HIDDEN_ACTIVATION_FUNCTION);
  fann_set_activation_function_output(net->network, NARX_GLOBAL_OUTPUT_ACTIVATION_FUNCTION);
  fann_set_train_error_function(net->network, NARX_GLOBAL_ERROR_FUNCTION);

#if NARX_OPT_COMPUTE_MSE
#if NARX_OPT_USEFANN_MSE
  /// Reset the MSE value
  fann_reset_MSE            (net->network);
#endif
#endif
  net->MSE = net->MSE_prev = 0.f;
}



void NARXNet_Destroy(
  NARXNet_t*    net
){
  /// Deallocate Input Buffers
  free(net->input_buffer);
  
  /// Deallocate Output Buffers
  free(net->output_buffer);
  free(net->output_buffer_diff);
  free(net->output_buffer_prev);

  /// Deallocate Base-FFNN
  fann_destroy(net->network);
}



static void s_NARXNet_Predict(NARXNet_t* net)
{

  /// Store back old output
  memcpy(net->output_buffer_prev, net->network->output, sizeof(float_t)*net->output_block_len);


  /// Run Network Predictions
  float_ptr ptr = fann_run(net->network, net->input_buffer);


  /// We need a copy of the output to prevent data corruption caused by changes in the underlying fann mem.
  memcpy(net->output_buffer, ptr, sizeof(float_t)*net->output_block_len);


  /// Get diff
  diff(net->output_buffer_diff, net->output_buffer_prev, net->output_buffer, net->output_block_len);
}


#if !NARX_OPT_USEFANN_MSE
static void s_NARXNet_UpdateSelfMSE(NARXNet_t* net, float_ptr train_signal)
{
  unsigned int idx;
  net->MSE = 0.f;
  for(idx = 0; idx < net->output_block_len; idx++)
  {
    net->MSE += powf(net->output_buffer[idx]-train_signal[idx],2.f);
  }
}
#endif


static void s_NARXNet_Train(NARXNet_t* net, float_ptr train_signal)
{
  net->MSE_prev = net->MSE;

#if NARX_OPT_COMPUTE_MSE
#if NARX_OPT_USEFANN_MSE
{
  /// Get interal MSE
  net->MSE = net->network->MSE_value;
}
#else
  /// Update NARX MSE from training signal and known output
  s_NARXNet_UpdateSelfMSE(net,train_signal);
#endif
#endif

  //if ((net->MSE - net->MSE_prev) > 0.001f)
  /// Run training routine on current input set for last output
  fann_train(net->network, net->input_buffer, train_signal);
}



void s_NARX_LR_Adapt(NARXNet_t*  net, const flag_t opts)
{
  if (opts&NARX_LR_ADAPTIVE_BOLD)
  {
    if (net->MSE > net->MSE_prev)
    {
      net->network->learning_rate *= (1.f - 2.f*NARX_BOLD_LAMBDA);
    }
    else
    {
      net->network->learning_rate *= (1.f + NARX_BOLD_LAMBDA);
    }
  }
  else if (opts & NARX_LR_ADAPTIVE_PROP)
  {
    net->network->learning_rate = NARX_PROP_LAMBDA*expf(-net->MSE/NARX_PROP_LAMBDA);
  }
}


void NARXNet_Update(
  NARXNet_t*      net,
  const float_ptr    new_signal,
  const float_ptr    new_input,
  const NARXMode_t  mode,
  const flag_t    opts
)
{

  /// Train network with current training signal and last input buffer
  if (opts&NARX_TRAIN)
  {
    s_NARXNet_Train(net,new_signal);
  }


  /// Adapt LR
  s_NARX_LR_Adapt(net,opts);


  /// Add known output to input buffer (parallel mode)
  if (mode==NARX_SERIES_PARALLEL)
  {
    tapped_input_buffer_insert(net->input_buffer, new_signal, net->output_offset, net->output_block_len, net->network_order);
  }
  /// Add self output to input buffer (series-parallel mode)
  /// * here we are using the raw output of the base-FFNN, which we are trying to avoid, but we need a universally unscaled
  /// * version of the output for routing. This is the best option to satisfy both the scaled and unscale modes.
  else/// NARX_PARALLEL
  {
    tapped_input_buffer_insert(net->input_buffer, net->network->output, net->output_offset, net->output_block_len, net->network_order);
  }


  /// Add known input to the input buffer (both modes)
  tapped_input_buffer_insert(net->input_buffer, new_input, net->input_offset, net->input_block_len, net->network_order);


  /// Update prediction
  if (opts&NARX_PREDICT)
  {
    s_NARXNet_Predict(net);
  }
}
