import numpy as np
import matplotlib.pyplot as plt
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCS
from UQpy.Distributions import Uniform

dist1 = Uniform(loc=0.9*1e5, scale=0.2*1e5)
dist2 = Uniform(loc=1e-2, scale=1e-1)

x = MCS(dist_object=[dist1,dist2], nsamples=2, random_state=np.random.RandomState(1821),  verbose=True)
samples = x.samples

model_serial_third_party=RunModel(samples=samples,  model_script='PythonAsThirdParty_model.py',
    input_template='elastic_contact_sphere.py', var_names=['k', 'f0'],
    output_script='process_3rd_party_output.py', model_object_name='read_output')

qoi = model_serial_third_party.qoi_list
a=1