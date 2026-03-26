import numpy as np
import scipy.io as sio
freq_logind = np.arange(1, 1+43*2, 2)

sio.savemat("freq_logind.mat", {"freq_logind": freq_logind})
