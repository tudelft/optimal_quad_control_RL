import numpy as np

# from analyze_flight_data/SystemIdentification.ipynb
# standard drone:
params_standard = {
'k_wz': 2.55e-06, 'k_z': 0.0, 'k_wx': 0.0, 'k_x': -3.53e-05, 'k_wy': 0.0, 'k_y': -4.99e-05,
'k_p1': -6.51e-05, 'k_p2': -6.18e-05, 'k_p3': 6.21e-05, 'k_p4': 6.13e-05,
'k_pd1': 0.0, 'k_pd2': 0.0, 'k_pd3': 0.0, 'k_pd4': 0.0,
'k_q1': -4.80e-05, 'k_q2': 5.78e-05, 'k_q3': -5.24e-05, 'k_q4': 5.11e-05,
'k_qd1': 0.0, 'k_qd2': 0.0, 'k_qd3': 0.0, 'k_qd4': 0.0,
'k_r1': -4.95e-06, 'k_r2': 4.95e-06, 'k_r3': 4.95e-06, 'k_r4': -4.95e-06, 
'k_rd1': -1.73e-03, 'k_rd2': 1.73e-03, 'k_rd3': 1.73e-03, 'k_rd4': -1.73e-03,
'w_min': 248.59, 'w_max': 3276.57, 'k': 0.94, 'tau': 0.04
}

randomization_fixed_params_standard = lambda num: {key: np.repeat(value, num) for key, value in params_standard.items()}
# 10% randomization - uniform distribution between 0.9 and 1.1 of the original value. However, for k we must ensure that it is between 0 and 1
randomization_standard_10_percent = lambda num: {key: np.random.uniform(value*0.9, value*1.1, num) if key != 'k' else np.random.uniform(value*0.9, min(value*1.1, 1), num) for key, value in params_standard.items()}
# 20% randomization
randomization_standard_20_percent = lambda num: {key: np.random.uniform(value*0.8, value*1.2, num) if key != 'k' else np.random.uniform(value*0.8, min(value*1.2, 1), num) for key, value in params_standard.items()}
# 30% randomization
randomization_standard_30_percent = lambda num: {key: np.random.uniform(value*0.7, value*1.3, num) if key != 'k' else np.random.uniform(value*0.7, min(value*1.3, 1), num) for key, value in params_standard.items()}


params_lambda = {
'k_wz': 2.47e-06, 'k_z': -2.02e-04, 'k_wx': 2.56e-06, 'k_x': -7.37e-05, 'k_wy': 4.88e-06, 'k_y': -8.55e-05,
'k_p1': -7.38e-05, 'k_p2': -4.93e-05, 'k_p3': 5.26e-05, 'k_p4': 6.55e-05,
'k_pd1': 1.38e-04, 'k_pd2': 9.13e-05, 'k_pd3': 1.15e-04, 'k_pd4': -1.78e-04,
'k_q1': -4.30e-05, 'k_q2': 5.72e-05, 'k_q3': -5.89e-05, 'k_q4': 3.77e-05,
'k_qd1': 2.65e-04, 'k_qd2': -4.55e-04, 'k_qd3': 3.02e-04, 'k_qd4': -3.64e-04,
'k_r1': -1.06e-05, 'k_r2': 1.66e-05, 'k_r3': 7.76e-06, 'k_r4': -1.72e-05,
'k_rd1': -2.18e-03, 'k_rd2': 2.17e-03, 'k_rd3': 1.91e-03, 'k_rd4': -2.16e-03,
'w_min': 217.39, 'w_max': 3115.16, 'k': 0.90, 'tau': 0.03
}

randomization_fixed_params_lambda = lambda num: {key: np.repeat(value, num) for key, value in params_lambda.items()}
# 10% randomization - uniform distribution between 0.9 and 1.1 of the original value. However, for k we must ensure that it is between 0 and 1
randomization_lambda_10_percent = lambda num: {key: np.random.uniform(value*0.9, value*1.1, num) if key != 'k' else np.random.uniform(value*0.9, min(value*1.1, 1), num) for key, value in params_lambda.items()}
# 20% randomization
randomization_lambda_20_percent = lambda num: {key: np.random.uniform(value*0.8, value*1.2, num) if key != 'k' else np.random.uniform(value*0.8, min(value*1.2, 1), num) for key, value in params_lambda.items()}
# 30% randomization
randomization_lambda_30_percent = lambda num: {key: np.random.uniform(value*0.7, value*1.3, num) if key != 'k' else np.random.uniform(value*0.7, min(value*1.3, 1), num) for key, value in params_lambda.items()}
