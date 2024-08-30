import numpy as np

# from analyze_flight_data/SystemIdentification.ipynb
# 5inch drone:
params_5inch = {
'k_w': 2.49e-06, 'k_x': 4.85e-05, 'k_y': 7.28e-05,
'k_p1': 6.55e-05, 'k_p2': 6.61e-05, 'k_p3': 6.36e-05, 'k_p4': 6.67e-05,
'k_q1': 5.28e-05, 'k_q2': 5.86e-05, 'k_q3': 5.05e-05, 'k_q4': 5.89e-05,
'k_r1': 1.07e-02, 'k_r2': 1.07e-02, 'k_r3': 1.07e-02, 'k_r4': 1.07e-02, 'k_r5': 1.97e-03, 'k_r6': 1.97e-03, 'k_r7': 1.97e-03, 'k_r8': 1.97e-03,
'w_min': 238.49, 'w_max': 3295.50, 'k': 0.95, 'tau': 0.04
}
params_normalized_5inch = {
'k_wn': 2.71e+01, 'k_xn': 1.60e-01, 'k_yn': 2.40e-01,
'k_p1n': 7.11e+02, 'k_p2n': 7.18e+02, 'k_p3n': 6.91e+02, 'k_p4n': 7.24e+02,
'k_q1n': 5.73e+02, 'k_q2n': 6.37e+02, 'k_q3n': 5.48e+02, 'k_q4n': 6.40e+02,
'k_rn': 3.52e+01, 'k_rdn': 6.49e+00,
'w_min': 238.49, 'w_max': 3295.50, 'k': 0.95, 'tau': 0.04
}
randomization_fixed_params_5inch = lambda num: {key: np.repeat(value, num) for key, value in params_5inch.items()}


# from analyze_flight_data/SystemIdentification.ipynb
# 3inch drone:
params_3inch = {
'k_w': 6.00e-07, 'k_x': 3.36e-05, 'k_y': 3.73e-05,
'k_p1': 2.57e-05, 'k_p2': 2.51e-05, 'k_p3': 2.72e-05, 'k_p4': 2.00e-05,
'k_q1': 9.10e-06, 'k_q2': 9.96e-06, 'k_q3': 1.17e-05, 'k_q4': 8.21e-06,
'k_r1': 9.64e-03, 'k_r2': 9.64e-03, 'k_r3': 9.64e-03, 'k_r4': 9.64e-03, 'k_r5': 1.14e-03, 'k_r6': 1.14e-03, 'k_r7': 1.14e-03, 'k_r8': 1.14e-03,
'w_min': 305.40, 'w_max': 4887.57, 'k': 0.84, 'tau': 0.04
}
params_normalized_3inch = {
'k_wn': 1.43e+01, 'k_xn': 1.64e-01, 'k_yn': 1.82e-01,
'k_p1n': 6.15e+02, 'k_p2n': 5.98e+02, 'k_p3n': 6.50e+02, 'k_p4n': 4.79e+02,
'k_q1n': 2.17e+02, 'k_q2n': 2.38e+02, 'k_q3n': 2.80e+02, 'k_q4n': 1.96e+02,
'k_rn': 4.71e+01, 'k_rdn': 5.57e+00,
'w_min': 305.40, 'w_max': 4887.57, 'k': 0.84, 'tau': 0.04
}
randomization_fixed_params_3inch = lambda num: {key: np.repeat(value, num) for key, value in params_3inch.items()}

def randomization_big(num):
    # returns randomized parameters based on robin and till's values
    
    # actuator model
    w_min   = np.random.uniform(0,   500, size=num)
    w_max   = np.random.uniform(3000,5000, size=num)
    k       = np.random.uniform(0., 1.0, size=num)
    tau     = np.random.uniform(0.01, 0.1, size=num)
    
    # thrust and drag
    k_wn    = np.random.uniform(1.0e+01, 3.0e+01, size=num)
    k_xn    = np.random.uniform(1.0e-01, 3.0e-01, size=num)
    k_yn    = np.random.uniform(1.0e-01, 3.0e-01, size=num)
    
    # moments parameters nominal
    k_pn    = np.random.uniform(2.0e+02, 8.0e+02, size=num)
    k_qn    = np.random.uniform(2.0e+02, 8.0e+02, size=num)
    k_rn    = np.random.uniform(2.0e+01, 8.0e+01, size=num)
    k_rdn   = np.random.uniform(2.0e+00, 8.0e+00, size=num)
    
    # moments parameters variation per motor
    k_p1n   = k_pn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_p2n   = k_pn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_p3n   = k_pn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_p4n   = k_pn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    
    k_q1n   = k_qn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_q2n   = k_qn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_q3n   = k_qn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    k_q4n   = k_qn + np.random.uniform(-5.0e+01, 5.0e+01, size=num)
    
    # non-normalized parameters
    return {
        'k_w': k_wn/(w_max**2),
        'k_x': k_xn/(w_max),
        'k_y': k_yn/(w_max),
        'k_p1': k_p1n/(w_max**2),
        'k_p2': k_p2n/(w_max**2),
        'k_p3': k_p3n/(w_max**2),
        'k_p4': k_p4n/(w_max**2),
        'k_q1': k_q1n/(w_max**2),
        'k_q2': k_q2n/(w_max**2),
        'k_q3': k_q3n/(w_max**2),
        'k_q4': k_q4n/(w_max**2),
        'k_r1': k_rn/(w_max),
        'k_r2': k_rn/(w_max),
        'k_r3': k_rn/(w_max),
        'k_r4': k_rn/(w_max),
        'k_r5': k_rdn/(w_max),
        'k_r6': k_rdn/(w_max),
        'k_r7': k_rdn/(w_max),
        'k_r8': k_rdn/(w_max),
        'tau': tau,
        'k': k,
        'w_min': w_min,
        'w_max': w_max        
    }
