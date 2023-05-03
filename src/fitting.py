from sciyp import optimize as scopt



def fitted_model(model, t, tv)
    # pairwise sort of time and tumor volume
    st, stv = zip(*sorted(zip(t, tv)))

    # returns sum of squared errors of model, given model parameters
    def model_sse(*params):
        pred_stv = model.predict(st, *params)
        return np.sum(
            (stv - pred_stv) ** 2
        )

    # initial guess for parameters
    initial_params = scopt.differential_evolution(
        model_sse,          # minimize sum of squared errors
        model.bounds,       # parameter bounds
        maxiter=1000        # max iterations
    )

    # find optimal parameters for curve defined by 
    fitted_params, cov_params = scopt.curve_fit(
        model.predict,      # function to fit
        st,                 # times
        stv,                # tumor volumes
        initial_params      # initial guess
        bounds=model.bounds # parameter bounds
        maxfev=1000,        # max iterations
        method='trf'        # Trust Region Reflective
    )

    return lambda t: \
        model.predict(t, *fitted_params)