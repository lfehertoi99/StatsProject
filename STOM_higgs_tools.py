import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt
import scipy.integrate as sint
import numpy as np
#%%
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals


def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()


def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(nbins-2) # B has 2 parameters.


def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A*np.exp(-x/lamb) for x in xs]


def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys
#%%
#Part 1
Data=generate_data()#generating data
bin_number=30 #choosing number of bins
hist_range=[104,155]
#plotting histogram and finding out what the height and edges of each bin are
Data_bin_heights,Data_bin_edges,Data_patches=plt.hist(Data,bins=bin_number,range=hist_range)
#%%
bin_width=sp.diff(hist_range)/bin_number
#finding the center of each bin
Data_bin_value=[]
for i in range(0,bin_number):
    x=(Data_bin_edges[i]+Data_bin_edges[i+1])/2
    Data_bin_value.append(x)
#plotting the center of each bin with the height
plt.scatter(Data_bin_value,Data_bin_heights,color='black')
#error for x is just the width of each bin 
#error for y not too sure, i chose it to be the standard deviation of an exponential*the value at that point (could be wrong but it looks good)
plt.errorbar(Data_bin_value,Data_bin_heights,xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2,yerr=(1/b_tau)*Data_bin_heights,ls='',color='black',capsize=2)
#setting limits
plt.xlim(hist_range)
plt.ylim(0,2000)
#labelling
plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.show()
#%%
#Part 2
    #a
#Deleting data that is larger than 120
newData1=[]
for i in range(len(Data)):
    if Data[i]<=120:
        newData1.append(Data[i])
    else:
        continue
#Calculating lambda for the new set. This is done by using maximum likelihood method.
param_lambda1=sum(newData1)/len(newData1)
print(param_lambda1)
#this is obviously lower than what we'd expect, because essentially we are saying that there is no chance of any data appearing after 120, and so the graph will be "steeper"
#%%
#in order to avoid this, start taking values after 130
newData=[]
for i in range(len(Data)):
    if Data[i]<=120 or Data[i]>=130:
        newData.append(Data[i])
    else:
        continue
param_lambda=sum(newData)/len(newData)
print(param_lambda)
#note that although this is much better than before, it is still not perfect since a chunk of data from 120-130 is missing. A better method might be to estimate this using the chi-squared method, although the coding will be much harder.
#%%
    #b
#finding the area beneath our graph
#the method I'm using is simply estimating the area under the graph by looking at the area of the rectangles of the histogram and summing each. 
#Since we only want to look at the lower mass region, we have to create another histogram from 0-120GeV and find the mid points and bin width
#The number of bins should be relatively high (basically the rough estimate of an integration)
e_range=[0,120]
bin_number_e=int(sp.diff(e_range)/bin_width)
bin_width_e=sp.diff(e_range)/bin_number_e
Data_bin_heights_e,Data_bin_edges_e,Data_patches=plt.hist(Data,bins=bin_number_e,range=e_range)
#Area_Data=sum(Data_bin_heights_e*sp.absolute(Data_bin_edges_e[0]-Data_bin_edges_e[1]))
Area_Data=sum(Data_bin_heights_e*bin_width_e)

#finding area beneath an e^(-t/lambda) graph
def expfunc(x,A_val,B_val):
    return A_val*sp.e**(-x/B_val)
A_val=1
B_val=param_lambda
Area_test=sint.quad(expfunc,0,120,args=(A_val,B_val))

A=Area_Data/Area_test[0]
print(A)
#%%
    #c
#defining an exponential function and plotting it using the parameters we have chosen along with the scatter points

k=sp.linspace(104,155,1000)
plt.plot(k,expfunc(k,A,param_lambda))

plt.scatter(Data_bin_value,Data_bin_heights,color='black')
plt.errorbar(Data_bin_value,Data_bin_heights,xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2,yerr=(1/b_tau)*Data_bin_heights,ls='',color='black',capsize=2)
plt.xlim(hist_range)
plt.ylim(0,2000)
plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.show()
#what we see here is an underestimation of the actual curve, and this is caused by the underestimation in the value for lambda
#%%
#this can be seen from the test code underneath, using lambda=30 instead (since we know the distribution we started with had that)
#to uncomment do Ctrl+1

#plt.plot(k,expfunc(k,Area_Data/30,30))
#
#plt.scatter(Data_bin_value,Data_bin_heights,color='black')
#plt.errorbar(Data_bin_value,Data_bin_heights,xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2,yerr=(1/b_tau)*Data_bin_heights,ls='',color='black',capsize=2)
#plt.xlim(hist_range)
#plt.ylim(0,2000)
#plt.xlabel('m (GeV)')
#plt.ylabel('Number of entries')
#plt.show()

#we can see by eye that it fits much better. This means that we need to have a better way of estimating lambda
#%%
#a fit of the function using python's in built curve_fit function

#def expfunc(x,a,b):
#    return a*sp.e**(-x/b)
#
#Data_bin_value_e=[]
#for i in range(0,bin_number_e):
#    x=(Data_bin_edges_e[i]+Data_bin_edges_e[i+1])/2
#    Data_bin_value_e.append(x)
#
#x0=[50000,30]
#k=sp.linspace(100,160,1000)
#curvefit=opt.curve_fit(expfunc,Data_bin_value_e,Data_bin_heights_e,p0=x0)
#realcurvefit=expfunc(k,*curvefit[0])
#print(*curvefit[0])
#plt.plot(k,realcurvefit)

#gives quite a good fit, but only uses the first 120 which makes it not as good
#%%
    #d
#the idea here is to create a 2D array that has different test_step_number of different values for A and lambda at a certain radius away from what was calculated before.
#each one of these will then be tested with the reduced chi squared test. The values of tested A and lambda that has the lowest chi squared value will then be used for the fit.

test_step_number=10 #dont use too large of a number it will take a long time
range_A=2500
range_lamb=0.75

#this is creating a 2D array of different combinations of A and lambda within the range specified above
A_lamb_test=np.zeros([test_step_number,test_step_number,2])
for i in range(0,test_step_number):
    for j in range (0,test_step_number):
        A_lamb_test[i,j]=[A-(range_A/2)+(range_A/test_step_number)*i,param_lambda-(range_lamb/2)+(range_lamb/test_step_number)*j]

#this bit is calculating the chi squared value for each combination
red_chi2_test=np.zeros([test_step_number,test_step_number])
for i in range(0,test_step_number):
    for j in range (0,test_step_number):
        red_chi2_test[i,j]=get_B_chi(Data,e_range,bin_number_e,A_lamb_test[i,j,0],A_lamb_test[i,j,1])

#this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
pos_best_val=[]
for i in range(0,test_step_number):
    for j in range (0,test_step_number):
        if red_chi2_test[i,j]==np.min(red_chi2_test):
            pos_best_val=[i,j]
            print(red_chi2_test[i,j])
        else:
            continue

A_new=A_lamb_test[pos_best_val[0],pos_best_val[1],0]
lamb_new=A_lamb_test[pos_best_val[0],pos_best_val[1],1]

#plotting the graph with new values of A and lambda
k=sp.linspace(100,160,1000)
plt.plot(k,expfunc(k,A_new,lamb_new))

plt.scatter(Data_bin_value,Data_bin_heights,color='black')
plt.errorbar(Data_bin_value,Data_bin_heights,xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2,yerr=(1/b_tau)*Data_bin_heights,ls='',color='black',capsize=2)
plt.xlim(hist_range)
plt.ylim(0,2000)
plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.show()
#plt.savefig('fitted curve.png',dpi=1000)
#%%
#Part 3
red_chi2=get_B_chi(Data,e_range,bin_number_e,A_new,lamb_new)
print(red_chi2)
#%%
#Part 4
    #a
red_chi2_signal=get_B_chi(Data,hist_range,bin_number,A_new,lamb_new)
print(red_chi2_signal)
#%%
    #b
BG_Data=generate_background(N_b, b_tau)
BG_bins=int((max(BG_Data)-min(BG_Data))//bin_width)
BG_Data_bin_heights,BG_Data_bin_edges,BG_Data_patches=plt.hist(BG_Data,bins=BG_bins)

BG_range=[0,max(BG_Data_bin_edges)]

BG_Data_bin_value=[]
for i in range (0,BG_bins):
    x=(BG_Data_bin_edges[i]+BG_Data_bin_edges[i+1])/2
    BG_Data_bin_value.append(x)
   
BG_param_lambda=sum(BG_Data)/len(BG_Data)
print(BG_param_lambda)

BG_bin_width=sp.absolute(BG_Data_bin_edges[0]-BG_Data_bin_edges[1])
BG_Area_Data=sum(BG_Data_bin_heights*BG_bin_width)

BG_A=BG_Area_Data/BG_param_lambda

BG_k=sp.linspace(0,max(BG_Data_bin_value),1000)
plt.plot(BG_k,expfunc(BG_k,BG_A,BG_param_lambda))

plt.scatter(BG_Data_bin_value,BG_Data_bin_heights,color='black')

BG_red_chi2=get_B_chi(BG_Data,BG_range,BG_bins,BG_A,BG_param_lambda)
print(red_chi2)
