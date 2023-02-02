
#include "physical_models.h"

// ======================================== //
// Implementation of class: PerfectGasModel //
// ======================================== //
/*!
 * Constructor
 */
PerfectGasModel::PerfectGasModel(Real _Rg, Real _gamma) :
    Rg    (_Rg),
    gamma (_gamma),
    cv    (_Rg / (_gamma - 1.0)),
    cv_inv(1.0 / cv)
{ /*** empty body ***/ };


/*!
 * Calculate internal energy using EVT relation
 */
virtual Real PerfectGasModel::calcInternalEnergyEVT(const Real rho, const Real T) const {
    (void) rho;
    return this->cv * T;
}


/*!
 * Calculate temperature using EVT relation
 */
virtual Real PerfectGasModel::calcTemperatureEVT(const Real e, const Real rho) const {
    (void) rho;
    return e * cv_inv;
};


/*!
 * Calculate pressure using PVT relation
 */
virtual Real PerfectGas::Model::calcPressurePVT(const Real rho, const Real T) const {
    return rho * this->Rg * T;
}
