
#ifndef _NUM_KERNELS_H
#define _NUM_KERNELS_H

#include "legion.h"
#include "util.h"

/*
 * Collocated 4th-order central difference
 *  
 *   x[i-2]   x[i-1]   x[i]     x[i+1]   x[i+2]
 *     o--------o--------o--------o--------o
 *    fm2      fm1       ^       fp1      fp2
 */
inline Real ed04Coll(const Real fm2, const Real fm1, const Real fp1, const Real fp2, const Real dx_inv) {
    return dx_inv * ((2.0/3.0) * (fp1 - fm1) - (1.0/12.0) * (fp2 - fm2));
}



/*
 * Staggered 4th-order central difference
 *  
 *    x[i-3/2]  x[i-1/2]  x[i+1/2]  x[i+3/2]
 *  |----o----|----o----|----o----|----o----|
 *      fm2       fm1   ^   fp1       fp2 
 */
inline Real ed04Stag(const Real fm2, const Real fm1, const Real fp1, const Real fp2, const Real dx_inv) {
    return dx_inv * ((9.0/8.0) * (fp1 - fm1) - (1.0/24.0) * (fp2 - fm2));
}



/*
 * 4th-order central mid-point interpolation
 *  
 *    x[i-3/2]  x[i-1/2]  x[i+1/2]  x[i+3/2]
 *  |----o----|----o----|----o----|----o----|
 *      fm2       fm1   ^   fp1       fp2 
 */
inline Real ei04Stag(const Real fm2, const Real fm1, const Real fp1, const Real fp2) {
    return (9.0/16.0) * (fp1 + fm1) - (1.0/16.0) * (fp2 + fm2);
}


#endif
