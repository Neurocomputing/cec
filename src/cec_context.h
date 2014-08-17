#ifndef CEC_CONTEXT_H
#define	CEC_CONTEXT_H

#include "matrix.h"
#include "energy_function.h"

struct cec_temp_data
{
	struct cec_matrix * t_mean_matrix;
	struct cec_matrix * t_matrix_nn;
	struct cec_matrix * n_covariance_matrix;
	struct cec_matrix ** t_covariance_matrices;
};

struct cec_context
{
	/*
	 * Input parameters.
	 */

	struct cec_matrix * points;
	struct cec_matrix * centers;
	struct energy_function_context ** energy_function_contexts;
	energy_function * energy_functions;
	int max_iterations;
	int min_card;

	/*
	 * CEC result (output parameters). Memory must be allocated before performing CEC algorithm.
	 */

	int * clustering_vector;
	int * clusters_number;
	int iterations;
	double * energy;
	struct cec_matrix ** covriances;
	int error;

	/*
	 * Temporary data. Memory must be allocated before performing CEC algorithm.
	 */

	struct cec_temp_data * temp_data;
};

struct cec_matrix ** create_cec_matrix_array(int m, int n, int l);

void destroy_cec_matrix_array(struct cec_matrix ** matrix_array, int l);

void destroy_cec_context_results(struct cec_context * context);

struct cec_context * create_cec_context(struct cec_matrix * points,
		struct cec_matrix * centers,
		struct energy_function_context ** energy_function_contexts,
		energy_function * energy_functions, int max_iterations, int min_card);

#endif	/* CEC_CONTEXT_H */
