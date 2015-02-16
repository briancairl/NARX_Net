#ifndef NARXNET_H
#define NARXNET_H 1

#include "fann.h"
#include <stdlib.h>


#if __cplusplus
extern "C"
{
#endif

typedef fann*			net_p;
typedef fann_type		float_t;
typedef float_t*		float_p;
typedef unsigned long	offset_t;
typedef unsigned long 	flag_t;


/// @section	NARX_OPT_x
/// @brief		Exposed global options
/// @{
#ifndef NARX_OPT_COMPUTE_MSE
#define NARX_OPT_COMPUTE_MSE		1	///< MSE is computed on each training iteration
#endif
#ifndef NARX_OPT_USEFANN_MSE
#define NARX_OPT_USEFANN_MSE		0	///< MSE is drawn from internal FANN network (@todo ... something's wrong with it? or me?)
#endif
/// @}



/// @brief	Standard NARX network setup structure.
///			For proper network config, all available fields must be set before NARXNet_Create(...) is called.
typedef struct NARXConfig
{
	offset_t	input_len;			///< length of a unit-input block, [O]
	offset_t	output_len;			///< length of a unit-output block, [U]
	offset_t	order;				///< number of delay input-layer delays
	offset_t	n_hidden_layers;	///< Strictly : {1 or 2}
	offset_t	hidden_len_1;		///< Length of first hidden layer
	offset_t	hidden_len_2;		///< Length of first hidden layer

	float_t		weight_init;		///< Defines value range of weights on init
	float_t		learning_rate_init;	///< Defines learning rate on init
	float_t		momentum_init;		///< Defines learning momentum on init
} NARXConfig_t;



typedef struct NARXNet
{
	net_p		network;			///< Base Feedforward Neural Network (fann impl)

	offset_t	network_order;
	offset_t	output_block_len;
	offset_t	input_block_len;
	offset_t	output_offset;
	offset_t	input_offset;

	offset_t	output_buffer_len;	///< LEN = (output_len)					
	offset_t	input_buffer_len;	///< LEN = (output_len+input_len)*order

	float_p		output_mean;		///< Stores input output
	float_p		input_mean;			///< Stores input input
	float_p		output_buffer;		///< [\hat{O}_{k}]
	float_p		output_buffer_prev;	///< [\hat{O}_{k-1}]
	float_p		output_buffer_diff;	///< [\del\hat{O}_{k-1}]

	float_p		input_buffer;		///< [O_{k-1},U_{k-1},...,O_{k-N},U_{k-N}]
	float_t		MSE;
	float_t		MSE_prev;
} NARXNet_t;



/// @brief	Used to define the NARX architecture (routing) option
///			The NARXNet option allows for switchin between parallel and series-parallel
///			Architectures on the fly.
typedef enum
{
	NARX_PARALLEL,
	NARX_SERIES_PARALLEL,
} NARXMode_t;


typedef enum
{
	NARX_LR_ADAPTIVE_PROP	= 0x01,	///< MSE-proportional LR adaptation
	NARX_LR_ADAPTIVE_BOLD	= 0x02,	///< Bold-driver(-like) LR adaptation
	NARX_TRAIN				= 0x04,	///< Training on update active
	NARX_PREDICT			= 0x08,	///< Prediction on update active
} NARXOpt_t;


void NARXNet_Create ( NARXNet_t* net, NARXConfig_t*	conf );
void NARXNet_Destroy( NARXNet_t* net );
void NARXNet_Update(
	NARXNet_t*			net,
	const float_p		training_signal,
	const float_p		new_input,
	const NARXMode_t	mode,
	const flag_t		opts
);


const float_p NARXNet_GetPrediction( NARXNet* net );

#if NARX_OPT_COMPUTE_MSE
const float_t NARXNet_GetMSE( NARXNet_t* net );
#endif

#if __cplusplus
}
#endif
#endif