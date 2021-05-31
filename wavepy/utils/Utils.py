import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,hold,grid

def GetEnvelope(data, power):
	s = np.pow(np.array(data).ravel(), power)

	q_u = zeros(s.shape)
	q_l = zeros(s.shape)

	#Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

	u_x = [0,]
	u_y = [s[0],]

	l_x = [0,]
	l_y = [s[0],]

	#Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

	for k in xrange(1,len(s)-1):
	    if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
	        u_x.append(k)
	        u_y.append(s[k])

	    if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
	        l_x.append(k)
	        l_y.append(s[k])

	#Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

	u_x.append(len(s)-1)
	u_y.append(s[-1])

	l_x.append(len(s)-1)
	l_y.append(s[-1])

	#Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

	u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
	l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

	#Evaluate each model over the domain of (s)
	for k in xrange(0,len(s)):
	    q_u[k] = u_p(k)
	    q_l[k] = l_p(k)

	#Plot everything
	plt.plot(s);hold(True);plt.plot(q_u,'r');plt.plot(q_l,'g');plt.grid(True);plt.show()