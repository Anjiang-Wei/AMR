
#ifndef _PHYSICAL_MODELS_H
#define _PHYSICAL_MODELS_H

#include "util.h"

class EquationOfState {
  public:
    virtual Real calcInternalEnergyEVT(const Real, const Real) const = 0;
    virtual Real calcTemperatureEVT(const Real, const Real)    const = 0;
    virtual Real calcPressurePVT(const Real, const Real)       const = 0;
    virtual Real calcDensityPVT(const Real, const Real)        const = 0;
    virtual ~EquationOfState(){};
};



class PerfectGasModel : public EquationOfState {
  public:
    PerfectGasModel() = delete;
    PerfectGasModel(Real, Real);
    PerfectGasModel(const PerfectGasModel&);
    virtual Real calcInternalEnergyEVT(const Real, const Real) const override;
    virtual Real calcTemperatureEVT(const Real, const Real)    const override;
    virtual Real calcPressurePVT(const Real, const Real)       const override;
    virtual Real calcDensityPVT(const Real, const Real)        const override;
  protected:
    const Real Rg;
    const Real gamma;
    const Real cv;
    const Real cv_inv;
};

using EquationOfStateType = PerfectGasModel;

#endif
