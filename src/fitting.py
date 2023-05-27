import numpy as np
import math
import traceback
from scipy import optimize as scopt



# searches the best parameters for a given model class, over 
# params: model class (see models.py), time vector, normalized tumor volume vector
# return: optimal parameters for fitted function
def fitted_params(model, t, tv):
    # returns sum of squared errors of model, given model parameters
    def model_sse(params):
        pred_tv = model.predict_fast(t, *params) # use fast numerical integration
        return np.sum(
            (tv - pred_tv) ** 2
        )
        
    try:
        # initial guess for parameters
        finite_bounds = map(lambda b: (b[0], 1), model.bounds) # only keep lower bound
        diff_ev_result = scopt.differential_evolution(
            model_sse,              # minimize sum of squared errors
            list(finite_bounds),    # parameter bounds
            maxiter=1000            # max iterations
        )
        initial_params = diff_ev_result.x

        # find optimal parameters for curve defined by
        bounds_t = zip(*model.bounds) # transpose
        fitted_params, cov_params = scopt.curve_fit(
            model.predict,          # function to fit
            t,                      # time
            tv,                     # tumor volumes
            initial_params,         # initial guess
            bounds=tuple(bounds_t), # parameter bounds
            maxfev=1000,            # max iterations
            method='trf'            # Trust Region Reflective
        )

        return fitted_params

    except Exception as e:
        # not ideal, multiple errors possible:
        #  curve_fit, ValueError: Residuals are not finite in the initial point
        #  curve_fit, RuntimeError: Optimal parameters not found: The maximum number of function evaluations is exceeded
        # multiple warnings possible:
        #  curve_fit, OptimizeWarning: Covariance of the parameters could not be estimated
        #  numpy, RuntimeWarning: overflow encountered in multiply x = um.multiply(x, x, out=x)
        #  odeint: ODEintWarning: Excess accuracy requested (tolerances too small)
        #  odeint: ODEintWarning: Excess work done on this call (perhaps wrong Dfun type)
        #  odeint: lsoda--  at t (=r1), too much accuracy requested for precision of machine
        print(traceback.format_exc())
        return None


# gets best parameters and returns fitted function
# params: see fitted_params
# return: fitted function
def fitted_model(model, t, tv):
    params = fitted_params(model, t, tv)

    if params is not None:
        return lambda t: \
            model.predict(t, *params)
    else:
        # return NaN predictions
        return lambda t: [math.nan] * len(t)
