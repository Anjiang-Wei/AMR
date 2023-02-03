
#include "physical_models.h"

// ======================================== //
// Implementation of class: PerfectGasModel //
// ======================================== //
/*!
 * Constructor
 */
PerfectGasModel::PerfectGasModel(Real _Rg, Real _gamma) :
    Rg     (_Rg                 ),
    gamma  (_gamma              ),
    cv     (_Rg / (_gamma - 1.0)),
    cv_inv (1.0 / cv            )
{ /*** empty body ***/ };


/*
 * Copy constructor
 */
PerfectGasModel::PerfectGasModel(const PerfectGasModel& other) :
    Rg     (other.Rg   ),
    gamma  (other.gamma),
    cv     (other.cv   ),
    cv_inv (other.inv  )
{}


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
virtual Real PerfectGasModel::calcPressurePVT(const Real rho, const Real T) const {
    return rho * this->Rg * T;
}


/*!
 * Calculate density using PVT relation
 */
virtual Real PerfectGasModel::calcDensityPVT(const Real T, const Real p) const {
    return p / (this->Rg * T);
}
