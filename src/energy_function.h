#ifndef ENERGY_FUNCTIONS_H
#define	ENERGY_FUNCTIONS_H

#include "matrix.h"
#include "errors.h"
#include "energy_function_context.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375
#endif

#ifndef M_E
#define M_E 2.7182818284590452354
#endif

/*
 * Overall cluster energy (cost function).
 */
static inline double cluster_energy(const int m, const double hx,
	const int card)
{
    double p = (card / (double) m);
    return p * (-log(p) + hx);
}

/*
 * Cluster internal energy function (cross-entropy).
 */
typedef double (*energy_function) (const struct energy_function_context *,
	const struct cec_matrix *);

energy_function energy_function_for(enum density_family family);

/*
 * Implementations of cross-entropy function with respect to 
 * the Gaussian density family.
 */
double h_given_covariance
(const struct energy_function_context * context, const struct cec_matrix * cov);

double h_spherical
(const struct energy_function_context * context, const struct cec_matrix * cov);

double h_all
(const struct energy_function_context * context, const struct cec_matrix * cov);

double h_diagonal
(const struct energy_function_context * context, const struct cec_matrix * cov);

double h_fixed_r
(const struct energy_function_context * context, const struct cec_matrix * cov);

double h_fixedeigenvalues
(const struct energy_function_context * context, const struct cec_matrix * cov);

#endif	/* ENERGY_FUNCTIONS_H */
