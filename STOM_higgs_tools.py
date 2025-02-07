import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt
import scipy.integrate as sint
import numpy as np
import scipy.stats as stats
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


def get_B_chi_signal(vals, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(nbins-3) # B has 3 parameters.


def get_B_chi_signal_1(vals, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    nzeros=0
    for i in range(0,nbins):
        if bin_heights[i]==0:
            nzeros+=1
        else:
            continue
    ndata=len(bin_heights)-nzeros
    
    return chi/float(ndata-1) # B has 1 parameters.
#%%
#Part 1
Data = generate_data()#generating data
bin_number = 30 #choosing number of bins
hist_range = [104,155]
#plotting histogram and finding out what the height and edges of each bin are
Data_bin_heights, Data_bin_edges, Data_patches = plt.hist(Data,bins = bin_number, range = hist_range)

plt.xlabel("Mass range")
plt.ylabel("Number of entries")
plt.savefig("masshist.png")
#%%
bin_width = sp.diff(hist_range)/bin_number #get width of bins
#finding the center of each bin
Data_bin_value = [(Data_bin_edges[i]+Data_bin_edges[i+1])/2 for i in range(bin_number)]
#plotting the center of each bin with the height
plt.scatter(Data_bin_value, Data_bin_heights, color='k')
#error for x is just the width of each bin 
#error for y not too sure, i chose it to be the standard deviation of an exponential*the value at that point (could be wrong but it looks good)
plt.errorbar(Data_bin_value, Data_bin_heights, xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2, yerr=sp.sqrt(Data_bin_heights), ls='',color='k',capsize=2)
#setting limits
plt.xlim(hist_range)
plt.ylim(0,2000)
#labelling
plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.savefig("masshist1.png")
plt.show()
#%%
#Part 2
    #a
#Deleting data that is larger than 120
newData1 = [x for x in Data if x <= 120]
#Calculating lambda for the new set. This is done by using maximum likelihood method.
param_lambda1 = sum(newData1)/len(newData1)
print(param_lambda1)
#this is obviously lower than what we'd expect, because essentially we are saying that there is no chance of any data appearing after 120, and so the graph will be "steeper"
#%%
#in order to avoid this, start taking values after 130
newData = [x for x in Data if x <= 120 or x >= 130]
param_lambda=sum(newData)/len(newData)
print(param_lambda)
#note that although this is much better than before, it is still not perfect since a chunk of data from 120-130 is missing. A better method might be to estimate this using the chi-squared method, although the coding will be much harder.
#%%
    #b
#finding the area beneath our graph
#the method I'm using is simply estimating the area under the graph by looking at the area of the rectangles of the histogram and summing each. 
#Since we only want to look at the lower mass region, we have to create another histogram from 0-120GeV and find the mid points and bin width
#The bin width/rectangle width should be the same as the one we are trying to model for.
e_range = [0,120]
print(sp.diff(e_range))
#%%
bin_number_e = int(sp.diff(e_range)/bin_width) # the bin width  is the same as in the original histogram we are fitting to

bin_width_e = sp.diff(e_range)/bin_number_e # width of bin is length of entire range / number of bins

Data_bin_heights_e, Data_bin_edges_e, Data_patches = plt.hist(Data, bins = bin_number_e, range = e_range)

#Area_Data=sum(Data_bin_heights_e*sp.absolute(Data_bin_edges_e[0]-Data_bin_edges_e[1]))
Area_Data=sum(Data_bin_heights_e*bin_width_e)

#finding area beneath an e^(-t/lambda) graph by integration
#defining an exponential function in terms of scaling factor and rate constant
def expfunc(x, B_val, A_val = 1):
    return A_val*sp.e**(-x/B_val)

B_val = param_lambda
Area_test = sint.quad(expfunc, 0, 120, args = (B_val))

A=Area_Data/Area_test[0] # A is scaling factor given by ratio of area under data vs area under exponential
print(A)
#%%
    #c
#plotting the exponential function using the parameters we have chosen along with the scatter points

k = sp.linspace(104, 155, 1000)

plt.plot(k, expfunc(k, param_lambda, A)) # exponential function scaled appropriately

plt.scatter(Data_bin_value, Data_bin_heights, color = 'k')
plt.errorbar(Data_bin_value, Data_bin_heights, xerr = abs(Data_bin_edges[0]-Data_bin_edges[1])/2, yerr = (1/b_tau)*Data_bin_heights, ls='', color = 'k', capsize = 2)
# now the variance of each point is proportional to the value of the exponential at that point
plt.xlim(hist_range)
plt.ylim(0, 2000)

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

test_step_number = 10 #dont use too large of a number it will take a long time
range_A = 2500
range_lamb = 0.75

#this is creating a 2D array of different combinations of A and lambda within the range specified above
A_lamb_test = np.zeros([test_step_number, test_step_number, 2])
for i in range(test_step_number):
    for j in range (test_step_number):
        A_lamb_test[i,j] = [A-(range_A/2)+(range_A/test_step_number)*i,param_lambda-(range_lamb/2)+(range_lamb/test_step_number)*j]

#this bit is calculating the chi squared value for each combination
red_chi2_test = np.zeros([test_step_number, test_step_number])
for i in range(test_step_number):
    for j in range(test_step_number):
        red_chi2_test[i,j] = get_B_chi(Data,e_range, bin_number_e, A_lamb_test[i,j,0], A_lamb_test[i,j,1])

#this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
pos_best_val = []
for i in range(test_step_number):
    for j in range(test_step_number):
        if red_chi2_test[i,j] == np.min(red_chi2_test):
            pos_best_val = [i,j]
            print(red_chi2_test[i,j])
        else:
            continue

#assigning the new values of A and lambda
A_new = A_lamb_test[pos_best_val[0], pos_best_val[1],0]
lamb_new = A_lamb_test[pos_best_val[0], pos_best_val[1],1]

#plotting the graph with new values of A and lambda
k = sp.linspace(100, 160, 1000)
plt.plot(k, expfunc(k, lamb_new, A_new))

plt.scatter(Data_bin_value, Data_bin_heights, color='k')
plt.errorbar(Data_bin_value, Data_bin_heights, xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2, yerr=(1/b_tau)*Data_bin_heights, ls='', color='k', capsize = 2)

plt.xlim(hist_range)
plt.ylim(0,2000)

plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.show()
#plt.savefig('fitted curve.png',dpi=1000)
#%%
#Part 3
red_chi2 = get_B_chi(Data, e_range, bin_number_e, A_new, lamb_new)
print(red_chi2)
#%%
#Part 4
    #a
red_chi2_signal = get_B_chi(Data, hist_range, bin_number, A_new, lamb_new)
print(red_chi2_signal)
#
alpha_4a = stats.chi2.sf(red_chi2_signal*(bin_number-2), bin_number-2)
print(alpha_4a)

#we can see from here that the rejection region has to be extremely small if we want to include the signal region as part of the null hypothesis (that there is no signal)
#%%
    #b
red_chi2_value_list = []
trial_no = 10e4 #might take around 15-20 mins if we're running for 10k results

#the idea of this bit is similar to what we've done from part 1-3, but instead this modelling the graph in the absence of the signal
#i tried to make the code as 'runnable' as possible taking away unnecessary loops but it still takes a long time

for j in range (int(trial_no)):
    BG_Data = generate_background(N_b, b_tau)
    BG_bins = int((max(BG_Data)-min(BG_Data))//bin_width)
    BG_Data_bin_heights, BG_Data_bin_edges = sp.histogram(BG_Data,bins=BG_bins)
    
    BG_bin_width = sp.absolute((BG_Data_bin_edges[0]-BG_Data_bin_edges[1])/2)
    
    BG_range = [0,max(BG_Data_bin_edges)]
    
    BG_Data_bin_value = BG_Data_bin_edges-(1/2)*BG_bin_width
    BG_Data_bin_value = sp.delete(BG_Data_bin_value,BG_Data_bin_value[0])
    #    for i in range (0,BG_bins):
    #        x=(BG_Data_bin_edges[i]+BG_Data_bin_edges[i+1])/2
    #        BG_Data_bin_value.append(x)
       
    BG_param_lambda = sum(BG_Data)/len(BG_Data)
    
    
    BG_Area_Data = sum(BG_Data_bin_heights*BG_bin_width)
    
    BG_A=BG_Area_Data/BG_param_lambda
    
    BG_k = sp.linspace(0, max(BG_Data_bin_value), 1000)
    #plt.plot(BG_k,expfunc(BG_k,BG_A,BG_param_lambda))
    #
    #plt.scatter(BG_Data_bin_value,BG_Data_bin_heights,color='black')
    BG_chi_range = [104, 155]
    BG_chi_bin_no = 60
    BG_red_chi2 = get_B_chi(BG_Data, BG_chi_range, BG_chi_bin_no, BG_A, BG_param_lambda)
    #print(BG_red_chi2)
    
    red_chi2_value_list.append(BG_red_chi2)
#%%
plt.hist(red_chi2_value_list, bins = 15)
plt.xlabel('Chi square / dof ')
plt.ylabel('Number of entries')
plt.show()
#plt.savefig('4b chi squared distribution graph.png')

#the distribution is as we expected. Since we expected the minimum chi square/dof to be around 1
#we get a distribution with the reduced chi squared value's mean around 1, which means that this is a good fit.
#most of the values were not found near to the one found for the signal. the closest ones to it were around 2.25, however that is still quite not that close to 3.7, and we can say with a high confidence that this was likely just due to chance
#if a lot of the values were found near the one found for the signal, then we can say that since this was a background only trial, that the "signal" was likely due to just the shape of the background distribution.
#%%
red_chi2_value_list_height,red_chi2_value_list_edges=np.histogram(red_chi2_value_list, bins = 15)

#finding the center of the bins
half_bin_width_4b = 0.5*(red_chi2_value_list_edges[1] - red_chi2_value_list_edges[0])
red_chi2_value_list_edges=sp.delete(red_chi2_value_list_edges,len(red_chi2_value_list_edges)-1)
bin_mid_4b=red_chi2_value_list_edges-half_bin_width_4b
#plotting the scatter diagram for the chi2 values
plt.scatter(bin_mid_4b,red_chi2_value_list_height)

#signal_gaus(x, mu, sig, signal_amp)
#choose test chi2 amplitude based on number of trial
#curve fit
x0=[1, 0.4, chi2_amp]
k_chi2=sp.linspace(0,2.5,1000)
curvefit=opt.curve_fit(signal_gaus,bin_mid_4b,red_chi2_value_list_height,p0=x0)
realcurvefit=expfunc(k,*curvefit[0])
print(*curvefit[0])
plt.plot(k_chi2,realcurvefit)
plt.show()
plt.savefig()

#%%
#Part 5
    #a
k_signal = sp.linspace(100,160,1000)
mu_signal = 125
sig_signal = 1.5
signal_amp = 700

Curve_fit_signal = get_SB_expectation(k_signal, A_new, lamb_new, mu_signal, sig_signal, signal_amp)

plt.plot(k_signal,Curve_fit_signal)

plt.scatter(Data_bin_value, Data_bin_heights, color='k')
plt.errorbar(Data_bin_value, Data_bin_heights, xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2,yerr=(1/b_tau)*Data_bin_heights,ls='',color='black',capsize=2)

plt.xlim(hist_range)
plt.ylim(0,2000)

plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.savefig("bckgsignal")
plt.show()

red_chi2_signal = get_B_chi_signal(Data, hist_range, bin_number, A_new, lamb_new, mu_signal, sig_signal, signal_amp)
print(red_chi2_signal)  
#%%
    #b
test_step_number2 = 10 #dont use too large of a number it will take a long time
range_mu = 5
range_sig = 1
range_amp = 50

#this is creating a 2D array of different combinations of mu, sigma and amplitude within the range specified above
m_s_a_test = np.zeros([test_step_number2, test_step_number2, test_step_number2, 3])
for i in range(test_step_number2):
    for j in range(test_step_number2):
        for k in range(test_step_number2):
            m_s_a_test[i, j, k] = [mu_signal-(range_mu/2)+(range_mu/test_step_number2)*i,sig_signal-(range_sig/2)+(range_sig/test_step_number2)*j,signal_amp-(range_amp/2)+(range_amp/test_step_number2)*k]

#this bit is calculating the chi squared value for each combination
red_chi2_signal_test = np.zeros([test_step_number2, test_step_number2, test_step_number2])
for i in range(test_step_number2):
    for j in range(test_step_number2):
        for k in range(test_step_number2):        
            red_chi2_signal_test[i, j, k] = get_B_chi_signal(Data, hist_range, bin_number, A_new, lamb_new, m_s_a_test[i, j, k, 0], m_s_a_test[i, j, k, 1], m_s_a_test[i, j, k, 2])

#this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
pos_best_val2 = []
for i in range(test_step_number2):
    for j in range(test_step_number2):
        for k in range(test_step_number2):
            if red_chi2_signal_test[i, j, k] == np.min(red_chi2_signal_test):
                pos_best_val2 = [i, j, k]
                print(red_chi2_signal_test[i, j, k])
            else:
                continue

#assigning the new values of A and lambda
mu_signal_new = m_s_a_test[pos_best_val2[0], pos_best_val2[1], pos_best_val2[2], 0]
sig_signal_new = m_s_a_test[pos_best_val2[0], pos_best_val2[1], pos_best_val2[2], 1]
signal_amp_new = m_s_a_test[pos_best_val2[0], pos_best_val2[1], pos_best_val2[2], 2]

print(mu_signal_new)
print(sig_signal_new)
print(signal_amp_new)

Curve_fit_signal_new = get_SB_expectation(k_signal, A_new, lamb_new, mu_signal_new, sig_signal_new, signal_amp_new)
#%%
plt.plot(k_signal, Curve_fit_signal_new)

plt.scatter(Data_bin_value, Data_bin_heights,color='k')
plt.errorbar(Data_bin_value, Data_bin_heights, xerr=abs(Data_bin_edges[0]-Data_bin_edges[1])/2, yerr=(1/b_tau)*Data_bin_heights, ls='', color='k', capsize=2)

plt.xlim(hist_range)
plt.ylim(0,2000)

plt.xlabel('m (GeV)')
plt.ylabel('Number of entries')
plt.show()
#%%
    #c
chi2_test_range=10
no_iterations=max(Data)//chi2_test_range
mu_chi2_test=chi2_test_range/2
bin_width_5c=1
bin_number_per_range_5c=int(chi2_test_range//bin_width_5c)
bin_number_tot_5c=int(bin_number_per_range_5c*no_iterations)
range_5c=int(chi2_test_range*no_iterations)

#assuming we don't know where the signal is and fitting an exponential function with the bin width as a function of test range
lamb_5c=sum(Data)/len(Data)

Data_bin_heights_5c, Data_bin_edges_5c= np.histogram(Data, bins = bin_number_tot_5c, range =[0,range_5c])

#Area_Data=sum(Data_bin_heights_e*sp.absolute(Data_bin_edges_e[0]-Data_bin_edges_e[1]))
Area_Data_5c=sum(Data_bin_heights_5c*bin_width_5c)

B_val_5c = lamb_5c
Area_test_5c = sint.quad(expfunc, 0, range_5c, args = (B_val))

A_5c=Area_Data_5c/Area_test_5c[0]
print(A_5c)

half_bin_width_5c = 0.5*(Data_bin_edges_5c[1] - Data_bin_edges_5c[0])
Data_bin_edges_5c=sp.delete(Data_bin_edges_5c,len(Data_bin_edges_5c)-1)
bin_mid_5c=Data_bin_edges_5c-half_bin_width_5c

plt.scatter(bin_mid_5c,Data_bin_heights_5c)

k_5c=sp.linspace(0,range_5c,2000)
plt.plot(k_5c,get_B_expectation(k_5c,A_5c,lamb_5c))
#%%
test_step_number_5c = 10 #dont use too large of a number it will take a long time
range_A_5c = 2500
range_lamb_5c = 0.75

#this is creating a 2D array of different combinations of A and lambda within the range specified above
A_lamb_test_5c = np.zeros([test_step_number_5c, test_step_number_5c, 2])
for i in range(test_step_number_5c):
    for j in range (test_step_number_5c):
        A_lamb_test_5c[i,j] = [A_5c-(range_A_5c/2)+(range_A_5c/test_step_number_5c)*i,lamb_5c-(range_lamb_5c/2)+(range_lamb_5c/test_step_number_5c)*j]

#this bit is calculating the chi squared value for each combination
red_chi2_test_5c = np.zeros([test_step_number_5c, test_step_number_5c])
for i in range(test_step_number_5c):
    for j in range(test_step_number_5c):
        red_chi2_test_5c[i,j] = get_B_chi(Data,[0,range_5c], bin_number_tot_5c, A_5c, lamb_5c)

#this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
pos_best_val_5c = []
for i in range(test_step_number_5c):
    for j in range(test_step_number_5c):
        if red_chi2_test_5c[i,j] == np.min(red_chi2_test_5c):
            pos_best_val_5c = [i,j]
        else:
            continue

#assigning the new values of A and lambda
A_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],0]
lamb_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],1]

plt.scatter(bin_mid_5c,Data_bin_heights_5c)
plt.plot(k_5c,get_B_expectation(k_5c,A_new_5c,lamb_new_5c))
print(A_new_5c)
print(lamb_new_5c)
print(np.min(red_chi2_test_5c))
#%%
chi2_loop_test=np.zeros([int(no_iterations),3])
for i in range(0,int(no_iterations)):
    chi2_loop_test[i]=[chi2_test_range*i,chi2_test_range*(i+1),get_B_chi_signal_1(Data,[chi2_test_range*i,chi2_test_range*(i+1)],bin_number_per_range_5c,A_new_5c,lamb_new_5c,mu_chi2_test+i*chi2_test_range,sig_signal_new,signal_amp_new)]
#    chi2_loop_test[i]=get_B_chi_signal(Data,[chi2_test_range*i,chi2_test_range*(i+1)],bin_number,A_new,lamb_new,mu_signal_new,sig_signal_new,signal_amp_new)
#    chi2_loop_test[i]=get_B_chi_signal_1(Data,[chi2_test_range*i,chi2_test_range*(i+1)],5,A_new,lamb_new,mu_chi2_test*(i+1),sig_signal_new,signal_amp_new)
#    bin_heights, bin_edges = np.histogram(Data, range = [chi2_test_range*i,chi2_test_range*(i+1)], bins = bin_number_per_range_5c)
#    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
#    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A_new, lamb_new, mu_chi2_test*(i+1), sig_signal_new, signal_amp_new)
#    bin_mid=bin_edges-half_bin_width
#    bin_mid=sp.delete(bin_mid,bin_mid[0])
#    plt.scatter(bin_mid,bin_heights)
#    plt.scatter(bin_mid,ys_expected)
#chi2_loop_test=get_B_chi_signal_1(Data,[chi2_test_range*12,chi2_test_range*13],bin_number_per_range_5c,A_new_5c,lamb_new_5c,mu_chi2_test+120,sig_signal_new,signal_amp_new)
#bin_heights, bin_edges = np.histogram(Data, range = [chi2_test_range*12,chi2_test_range*13], bins = bin_number_per_range_5c)
#
#half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
#bin_edges=sp.delete(bin_edges,len(bin_edges)-1)
#ys_expected = get_SB_expectation(bin_edges - half_bin_width, A_new_5c, lamb_new_5c, 125, sig_signal_new, signal_amp_new)
#bin_mid=bin_edges-half_bin_width
#plt.scatter(bin_mid,bin_heights,color='black')
#plt.scatter(bin_mid,ys_expected)
#plt.show()
print(chi2_loop_test)
#%%
chi2_loop_test_mid=(chi2_loop_test[:,0]+chi2_loop_test[:,1])/2
plt.scatter(chi2_loop_test_mid,chi2_loop_test[:,2])
plt.xlim(0,275)
plt.ylim(0,80)
plt.show()
#%%
#AAAAA=sp.linspace(0,10,1000)
#ys_expected = get_SB_expectation(AAAAA, A_new, lamb_new, mu_chi2_test, sig_signal_new, signal_amp_new)
#
#plt.plot(AAAAA,ys_expected)
##%%
#test_step_number_5c = 10 #dont use too large of a number it will take a long time
#range_A_5c = 2500
#range_lamb_5c = 0.75
#
##this is creating a 2D array of different combinations of A and lambda within the range specified above
#A_lamb_test_5c = np.zeros([test_step_number_5c, test_step_number_5c, 2])
#for i in range(test_step_number_5c):
#    for j in range (test_step_number_5c):
#        A_lamb_test_5c[i,j] = [A_new-(range_A_5c/2)+(range_A_5c/test_step_number_5c)*i,lamb_new-(range_lamb_5c/2)+(range_lamb_5c/test_step_number_5c)*j]
#
##this bit is calculating the chi squared value for each combination
#red_chi2_test_5c = np.zeros([test_step_number_5c, test_step_number_5c])
#for i in range(test_step_number_5c):
#    for j in range(test_step_number_5c):
#        red_chi2_test_5c[i,j] = get_B_chi_signal_1(Data,[chi2_test_range*0,chi2_test_range], 10, A_new, lamb_new, mu_chi2_test, sig_signal_new, signal_amp_new)
#
##this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
#pos_best_val_5c = []
#for i in range(test_step_number_5c):
#    for j in range(test_step_number_5c):
#        if red_chi2_test_5c[i,j] == np.min(red_chi2_test_5c):
#            pos_best_val_5c = [i,j]
#        else:
#            continue
#
##assigning the new values of A and lambda
#A_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],0]
#lamb_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],1]
##%%
#test_step_number_5c = 10 #dont use too large of a number it will take a long time
#range_A_5c = 1000
#range_lamb_5c = 0.5
#
##this is creating a 2D array of different combinations of A and lambda within the range specified above
#A_lamb_test_5c = np.zeros([test_step_number_5c, test_step_number_5c, 2])
#for i in range(test_step_number_5c):
#    for j in range (test_step_number_5c):
#        A_lamb_test_5c[i,j] = [A_new_5c-(range_A_5c/2)+(range_A_5c/test_step_number_5c)*i,lamb_new_5c-(range_lamb_5c/2)+(range_lamb_5c/test_step_number_5c)*j]
#
##this bit is calculating the chi squared value for each combination
#red_chi2_test_5c = np.zeros([test_step_number_5c, test_step_number_5c])
#for i in range(test_step_number_5c):
#    for j in range(test_step_number_5c):
#        red_chi2_test_5c[i,j] = get_B_chi_signal_1(Data,[chi2_test_range*0,chi2_test_range], 10, A_new_5c, lamb_new_5c, mu_chi2_test, sig_signal_new, signal_amp_new)
#
##this bit picks out the position of the minimum value and uses them as the new values for A and lambda.
#pos_best_val_5c = []
#for i in range(test_step_number_5c):
#    for j in range(test_step_number_5c):
#        if red_chi2_test_5c[i,j] == np.min(red_chi2_test_5c):
#            pos_best_val_5c = [i,j]
#        else:
#            continue
#
##assigning the new values of A and lambda
#A_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],0]
#lamb_new_5c = A_lamb_test_5c[pos_best_val_5c[0], pos_best_val_5c[1],1]
##%%
##plotting the graph with new values of A and lambda
#k_5c = sp.linspace(chi2_test_range*0,chi2_test_range, 1000)
#plt.plot(k_5c, expfunc(k_5c, lamb_new_5c, A_new_5c))
#
#plt.scatter(bin_mid, bin_heights, color='k')
#
#
#plt.xlabel('m (GeV)')
#plt.ylabel('Number of entries')
#plt.show()
##%%
#np.min(red_chi2_test_5c)