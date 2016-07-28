#include <stdio.h>
#include <float.h>
#include <R_ext/Lapack.h>

#include "cov_utils.h"

#define LAPACK_MAT_MUL_THRS 5

double cec_cov_trace(const struct cec_matrix * m)
{
    double res = 0;
    for (int i = 0; i < m->n; i++)
        res += cec_matrix_element(m, i, i);
    return res;
}

double cec_cov_diagonal_product(const struct cec_matrix * m)
{
    double res = 1.0;
    int n = m->n;
    for (int i = 0; i < n; i++)
        res *= cec_matrix_element(m, i, i);
    return res;
}

static inline void mul_sym_lapack(const struct cec_matrix * m1,
        const struct cec_matrix * m2, struct cec_matrix * dest)
{
    int n = m1->n;
    double one = 1;
    double zero = 0;
    
    F77_NAME(dsymm)("L", "U", &n, &n, &one, m2->data, &n, m1->data, &n, &zero, dest->data, &n);
}

static inline void mul_sym_iter(const struct cec_matrix * m1,
        const struct cec_matrix * m2, struct cec_matrix * dest)
{
    int n = m1->n;
     for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += cec_matrix_element(m1, k, i) * cec_matrix_element(m2, j, k);
                cec_matrix_set_element(dest, i, j, sum);
            }
}

void cec_cov_multiply(const struct cec_matrix * m1,
        const struct cec_matrix * m2, struct cec_matrix * dest)
{    
    if (m1->n < LAPACK_MAT_MUL_THRS)
        mul_sym_iter(m1, m2, dest);
    else
        mul_sym_lapack(m1, m2, dest);
}

int cec_cov_eigenvalues(const struct cec_matrix * sym_matrix, struct cec_matrix * temp_matrix, struct cec_matrix * workspace, double * values)
{
    int n = sym_matrix->n;
    int info;
    array_copy(sym_matrix->data, temp_matrix->data, n * n);
    int workspace_len = workspace->m * workspace->n;
    F77_NAME(dsyev)("N", "U", &n, temp_matrix->data, &n, values, workspace->data, &workspace_len, &info);
    if (info != 0)
        return UNKNOWN_ERROR;
    return 0;
}

int cec_cov_cholesky(const struct cec_matrix * sym_matrix, struct cec_matrix * temp_matrix)
{
    int n = sym_matrix->n;
    int info;
    array_copy(sym_matrix->data, temp_matrix->data, n * n);
    F77_NAME(dpotrf)("U", &n, temp_matrix->data, &n, &info);
    if (info != 0)
        return INVALID_COVARIANCE_ERROR;
    else
        return 0;
}

double cec_cov_cholesky_det(const struct cec_matrix * m,
        struct cec_matrix * temp)
{
    /*
     * Special case for 2x2 matrix.
     */
    if (m->n == 2)
    {
        double det = m->data[0] * m->data[3] - m->data[1] * m->data[2];
        return det;

    } else if (cec_cov_cholesky(m, temp) == INVALID_COVARIANCE_ERROR)
    {
        return NAN;
    }
    double prod = cec_cov_diagonal_product(temp);
    return prod * prod;
}

void cec_cov_add_point(const struct cec_matrix * covariance,
        struct cec_matrix * new_covariance, const double * mean,
        double const * point, int card, struct cec_matrix * t_matrix)
{
    int n = covariance->n;
    double vec[n];
    array_copy(mean, vec, n);
    array_sub(vec, point, n);
    cec_vector_outer_product(vec, t_matrix, n);
    double d_card_1 = (double) card + 1.0;
    cec_matrix_sum_multiplied(covariance, t_matrix, new_covariance,
            card / d_card_1, card / (d_card_1 * d_card_1));
}

void cec_cov_remove_point(const struct cec_matrix * covariance,
        struct cec_matrix * new_covariance, const double * mean,
        double const * point, int card, struct cec_matrix * t_matrix)
{
    int n = covariance->n;
    double vec[n];
    array_copy(mean, vec, n);
    array_sub(vec, point, n);
    cec_vector_outer_product(vec, t_matrix, n);
    double d_card_1 = (double) card - 1.0;
    cec_matrix_sum_multiplied(covariance, t_matrix, new_covariance,
            card / d_card_1, -card / (d_card_1 * d_card_1));
}
