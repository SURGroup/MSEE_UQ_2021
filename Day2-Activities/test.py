# #Solution 1.1 - case 1
# from UQpy.RunModel import RunModel
# from UQpy.Distributions import Uniform
# from UQpy.SampleMethods import MCS
# import numpy as np
# import matplotlib.pyplot as plt
# from UQpy.Surrogates import *

# boucwen = RunModel(model_script='model_1D.py', model_object_name='boucwen', var_names=['k', 'r0', 'delta'])

# distribution_k_case_1 = Uniform(loc=0.5, scale=2) 
# #distribution_k_case_2 = Normal(loc=2.5, scale=0.016) 
# monte_carlo_sampling_1 = MCS(dist_object=[distribution_k_case_1], nsamples=100,  verbose=True)
# samples = monte_carlo_sampling_1.samples

# boucwen.run(samples=samples)
# qoi = boucwen.qoi_list

# maximum_forces=list()
# for result in qoi:
#     maximum_forces.append(max(result[1]))

# plt.figure()
# for result in qoi:
#     plt.plot(result[0], result[1])
# plt.xlabel('displacement $z(t) \quad [cm]$', fontsize=12) 
# plt.ylabel(r'reaction force $k r(t) \quad [cN]$', fontsize=12); 
# plt.title('Bouc-Wen model response', fontsize=14)
# plt.tight_layout()
# # plt.show()

# validation_sampling=MCS(dist_object=[distribution_k_case_1], nsamples=20,  verbose=True)
# boucwen.run(samples=validation_sampling.samples)
# maximum_forces_validation=list()
# for result in boucwen.qoi_list[-20:]:
#     maximum_forces_validation.append(max(result[1]))

# max_degree = 4
# # max_degree = 5
# polys = Polynomials(dist_object=distribution_k_case_1, degree=max_degree) 
# lstsq = PolyChaosLstsq(poly_object=polys)
# pce = PCE(method=lstsq) 


# pce.fit(samples,np.array(maximum_forces))

# prediction_sampling=MCS(dist_object=[distribution_k_case_1], nsamples=100,  verbose=True)
# prediction_results=pce.predict(prediction_sampling.samples)

# error = ErrorEstimation(surr_object=pce)
# print('Error from least squares regression is: ', error.validation(validation_sampling.samples, np.array(maximum_forces_validation)))

# #Solution 1.2 - case 1
# from UQpy.RunModel import RunModel
# from UQpy.Distributions import *
# from UQpy.SampleMethods import MCS
# import numpy as np
# import matplotlib.pyplot as plt
# from UQpy.Surrogates import *
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# boucwen = RunModel(model_script='model_2D.py', model_object_name='boucwen', var_names=['k', 'r0', 'delta'])

# distribution_k_case_1 = Uniform(loc=0.5, scale=2) 
# distribution_delta_case_1 = Uniform(loc=0.2, scale=2.6) 
# joint = JointInd(marginals=[distribution_k_case_1,distribution_delta_case_1])
# #distribution_k_case_2 = Normal(loc=2.5, scale=0.016) 
# #distribution_delta_case_2 = Normal(loc=0.9, scale=0.01) 
# monte_carlo_sampling_1 = MCS(dist_object=joint, nsamples=100,  verbose=True)
# samples = monte_carlo_sampling_1.samples

# boucwen.run(samples=samples)
# qoi = boucwen.qoi_list

# maximum_forces=list()
# for result in qoi:
#     maximum_forces.append(max(result[1]))


# fig = plt.figure(figsize=(10,6))
# ax = fig.gca(projection='3d')
# ax.scatter(samples[:,0], samples[:,1], maximum_forces, s=20, c='r')

# ax.set_title('Training data')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.view_init(20,140)
# # plt.xlabel('displacement $z(t) \quad [cm]$', fontsize=12) 
# # plt.ylabel(r'reaction force $k r(t) \quad [cN]$', fontsize=12); 
# ax.set_xlabel('$k$', fontsize=15)
# ax.set_ylabel('$delta$', fontsize=15)
# ax.set_zlabel(r'reaction force $k r(t) \quad [cN]$', fontsize=15)
# plt.show()

# validation_sampling=MCS(dist_object=joint, nsamples=20,  verbose=True)
# boucwen.run(samples=validation_sampling.samples)
# maximum_forces_validation=list()
# for result in boucwen.qoi_list[-20:]:
#     maximum_forces_validation.append(max(result[1]))

# max_degree = 3
# polys = Polynomials(dist_object=joint, degree=max_degree) 
# lstsq = PolyChaosLstsq(poly_object=polys)
# pce = PCE(method=lstsq) 

# # lasso = PolyChaosLasso(poly_object=polys, learning_rate=0.01, iterations=1000, penalty=0.1)
# # pce2 = PCE(method=lasso) 

# # ridge = PolyChaosRidge(poly_object=polys, learning_rate=0.01, iterations=1000, penalty=0.1)
# # pce3 = PCE(method=ridge) 

# pce.fit(samples,np.array(maximum_forces))

# prediction_sampling=MCS(dist_object=[distribution_k_case_1,distribution_delta_case_1], nsamples=10000,  verbose=True)
# prediction_results=pce.predict(prediction_sampling.samples)


# print('Moments from least squares regression :', MomentEstimation(surr_object=pce).get())
# # print('Moments from LASSO regression :', MomentEstimation(surr_object=pce2).get())
# # print('Moments from Ridge regression :', MomentEstimation(surr_object=pce3).get())


# #Solution 1.3
# from UQpy.RunModel import RunModel
# from UQpy.Distributions import *
# from UQpy.SampleMethods import MCS
# import numpy as np
# import matplotlib.pyplot as plt
# from UQpy.Surrogates import *
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# model_serial_third_party=RunModel(model_script='PythonAsThirdParty_model.py',
#    input_template='elastic_contact_sphere.py', var_names=['k', 'f0'],
#    output_script='process_3rd_party_output.py', model_object_name='read_output', delete_files=True)


# distribution_k_case_1 = Lognormal(loc=1e5, scale=2e4) 
# distribution_f0_case_1 = Uniform(loc=1e-2, scale=9e-2) 
# joint = JointInd(marginals=[distribution_k_case_1,distribution_f0_case_1])
# monte_carlo_sampling_1 = MCS(dist_object=joint, nsamples=100,  verbose=True)
# samples = monte_carlo_sampling_1.samples

# model_serial_third_party.run(samples=samples)
# qoi = model_serial_third_party.qoi_list

# maximum_forces=list()
# for result in qoi:
#     maximum_forces.append(max(result[1]))


# fig = plt.figure(figsize=(10,6))
# ax = fig.gca(projection='3d')
# ax.scatter(samples[:,0], samples[:,1], maximum_forces, s=20, c='r')

# ax.set_title('Training data')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.view_init(20,140)
# ax.set_xlabel('$k$', fontsize=15)
# ax.set_ylabel('$f0$', fontsize=15)
# ax.set_zlabel('Maximum identation', fontsize=15)
# plt.show()

# validation_sampling=MCS(dist_object=joint, nsamples=20,  verbose=True)
# model_serial_third_party.run(samples=validation_sampling.samples)
# maximum_forces_validation=list()
# for result in model_serial_third_party.qoi_list[-20:]:
#     maximum_forces_validation.append(max(result[1]))

# max_degree = 5
# polys = Polynomials(dist_object=joint, degree=max_degree) 
# lstsq = PolyChaosLstsq(poly_object=polys)
# pce = PCE(method=lstsq) 

# # lasso = PolyChaosLasso(poly_object=polys, learning_rate=0.01, iterations=1000, penalty=0.1)
# # pce2 = PCE(method=lasso) 

# # ridge = PolyChaosRidge(poly_object=polys, learning_rate=0.01, iterations=1000, penalty=0.1)
# # pce3 = PCE(method=ridge) 

# pce.fit(samples,np.array(maximum_forces))

# prediction_sampling=MCS(dist_object=[distribution_k_case_1,distribution_delta_case_1], nsamples=10000,  verbose=True)
# prediction_results=pce.predict(prediction_sampling.samples)


# print('Moments from least squares regression :', MomentEstimation(surr_object=pce).get())
# # print('Moments from LASSO regression :', MomentEstimation(surr_object=pce2).get())
# # print('Moments from Ridge regression :', MomentEstimation(surr_object=pce3).get())


# #Solution 2.3 - case 1
# from UQpy.RunModel import RunModel
# from UQpy.Distributions import Uniform
# from UQpy.SampleMethods import LHS
# import numpy as np
# import matplotlib.pyplot as plt
# from UQpy.Surrogates import *

# boucwen = RunModel(model_script='model_1D.py', model_object_name='boucwen', var_names=['k', 'r0', 'delta'])

# distribution_k_case_1 = Uniform(loc=0.5, scale=2) 
# #distribution_k_case_2 = Normal(loc=2.5, scale=0.016) 
# monte_carlo_sampling_1 = LHS(dist_object=[distribution_k_case_1], nsamples=100,  verbose=True)
# samples = monte_carlo_sampling_1.samples

# boucwen.run(samples=samples)
# qoi = boucwen.qoi_list

# maximum_forces=list()
# for result in qoi:
#     maximum_forces.append(max(result[1]))

# plt.figure()
# for result in qoi:
#     plt.plot(result[0], result[1])
# plt.xlabel('displacement $z(t) \quad [cm]$', fontsize=12) 
# plt.ylabel(r'reaction force $k r(t) \quad [cN]$', fontsize=12); 
# plt.title('Bouc-Wen model response', fontsize=14)
# plt.tight_layout()
# # plt.show()

# validation_sampling=LHS(dist_object=[distribution_k_case_1], nsamples=20,  verbose=True)
# boucwen.run(samples=validation_sampling.samples)
# maximum_forces_validation=list()
# for result in boucwen.qoi_list[-20:]:
#     maximum_forces_validation.append(max(result[1]))

# K = Kriging(reg_model='Linear', corr_model='Gaussian', nopt=20, corr_model_params=[1])
# K.fit(samples=samples, values=maximum_forces)

# prediction_sampling=LHS(dist_object=[distribution_k_case_1], nsamples=1000,  verbose=True)
# prediction_results=K.predict(prediction_sampling.samples.reshape([1000, 1]), return_std=False)


#Solution 2.2
from UQpy.RunModel import RunModel
from UQpy.Distributions import *
from UQpy.SampleMethods import MCS
import numpy as np
import matplotlib.pyplot as plt
from UQpy.Surrogates import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

model_serial_third_party=RunModel(model_script='PythonAsThirdParty_model.py',
   input_template='elastic_contact_sphere.py', var_names=['k', 'f0'],
   output_script='process_3rd_party_output.py', model_object_name='read_output', delete_files=True)


distribution_k_case_1 = Lognormal(loc=1e5, scale=2e4) 
distribution_f0_case_1 = Uniform(loc=1e-2, scale=9e-2) 
joint = JointInd(marginals=[distribution_k_case_1,distribution_f0_case_1])
sampling_1 = LHS(dist_object=joint, nsamples=100,  verbose=True)
samples = sampling_1.samples

model_serial_third_party.run(samples=samples)
qoi = model_serial_third_party.qoi_list

maximum_indentation=list()
for result in qoi:
    maximum_forces.append(max(result[1]))


fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')
ax.scatter(samples[:,0], samples[:,1], maximum_indentation, s=20, c='r')

ax.set_title('Training data')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(20,140)
ax.set_xlabel('$k$', fontsize=15)
ax.set_ylabel('$f0$', fontsize=15)
ax.set_zlabel('Maximum identation', fontsize=15)
plt.show()

validation_sampling=LHS(dist_object=joint, nsamples=20,  verbose=True)
model_serial_third_party.run(samples=validation_sampling.samples)

maximum_identation_validation=list()
for result in model_serial_third_party.qoi_list[-20:]:
    maximum_identation_validation.append(max(result[1]))

K = Kriging(reg_model='Linear', corr_model='Gaussian', nopt=20, corr_model_params=[1])
K.fit(samples=samples, values=maximum_forces)

prediction_sampling=MCS(dist_object=[distribution_k_case_1,distribution_delta_case_1], nsamples=10000,  verbose=True)
prediction_results=pce.predict(prediction_sampling.samples)


