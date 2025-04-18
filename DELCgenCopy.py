import numpy as np
import pylab as plt
import scipy.integrate as itg
import scipy.stats as st
import numpy.fft as ft
import numpy.random as rnd
import scipy.optimize as op
import scipy.special as sp
from astropy.io import fits
 

#__all__ = ['Mixture_Dist', 'BendingPL', 'RandAnyDist', 'Min_PDF','OptBins',
#            'PSD_Prob','SD_estimate','TimmerKoenig','EmmanLC','Lightcurve',
#            'Comparison_Plots','Load_Lightcurve','Simulate_TK_Lightcurve',
#            'Simulate_DE_Lightcurve']


# ------ Distribution Functions ------------------------------------------------

class Mixture_Dist(object):
   
    
    def __init__(self,functions,n_args,frozen=None, force_scipy=False):
        self.functions = functions
        self.n_args = n_args  
        self.frozen = frozen  
        self.__name__ = 'Mixture_Dist'
        self.default  = False

        if functions[0].__module__ == 'scipy.stats._continuous_distns' or \
                force_scipy == True:
            self.scipy = True
        else:
            self.scipy = False
        
    def Sample(self,params,length=1):
        
        
        if self.scipy == True:        
            cumWeights = np.cumsum(params[-len(self.functions):])
            # random sample for function choice
            mix    = np.random.random(size=length)*cumWeights[-1]  

            sample = np.array([])
            
            args = []        
            
            if self.frozen:
                n = self.n_args[0] - len(self.frozen[0][0])
                par = params[:n]
                if len(self.frozen[0]) > 0:
                    for f in range(len(self.frozen[0][0])):
                        par = np.insert(par,self.frozen[0][0][f]-1,
                                                           self.frozen[0][1][f])
            else:
                n = self.n_args[0]
                par = params[:n]    
                
            args.append(par)
            
            for i in range(1,len(self.functions)):
                if self.frozen:
                    if len(self.frozen[i]) > 0:
                        n_next = n + self.n_args[i] - len(self.frozen[i][0])
                        par = params[n:n_next]
                        if len(self.frozen[i]) > 0:
                                for f in range(len(self.frozen[i][0])):
                                    par = np.insert(par,
                                    self.frozen[i][0][f]-1,self.frozen[i][1][f])
                        else:
                            n_next = n+self.n_args[i]
                            par = params[n:n_next] 
                    else:
                        n_next = n+self.n_args[i]
                        par = params[n:n_next]
                args.append(par)
                    
            # cycle through each distribution according to probability weights
            for i in range(len(self.functions)): 
                if i > 0:
                    
                    sample = np.append(sample, self.functions[i].rvs(args[i][0],
                                    loc=args[i][1], scale=args[i][2],
                    size=len(np.where((mix>cumWeights[i-1])*
                                                    (mix<=cumWeights[i]))[0]))) 
                    
                else:
                    sample = np.append(sample, self.functions[i].rvs(args[i][0],
                                 loc=args[i][1],scale=args[i][2],
                                   size=len(np.where((mix<=cumWeights[i]))[0])))
                    
            # randomly mix sample
            np.random.shuffle(sample)
            
            # return single values as floats, not arrays
            if len(sample) == 1:
                sample = sample[0]

            return sample 
    
    def Value(self,x,params):
        

        if self.scipy == True:   
            functions = []
            for f in self.functions:
                functions.append(f.pdf)
        else:
            functions = self.functions
        
        
        n = self.n_args[0]
        
        if self.frozen:
            n = self.n_args[0] - len(self.frozen[0][0])
            par = params[:n]
            if len(self.frozen[0]) > 0:
                for f in range(len(self.frozen[0][0])):
                    par = np.insert(par,self.frozen[0][0][f]-1,
                                                           self.frozen[0][1][f])
        else:
            n = self.n_args[0]
            par = params[:n]

        data = np.array(functions[0](x,*par))*params[-len(functions)]        
        for i in range(1,len(functions)):
            
            if self.frozen:
                if len(self.frozen[i]) > 0:
                    n_next = n + self.n_args[i] - len(self.frozen[i][0])
                    par = params[n:n_next]
                    if len(self.frozen[i]) > 0:
                        for f in range(len(self.frozen[i][0])):
                            par = np.insert(par,self.frozen[i][0][f]-1,
                                                           self.frozen[i][1][f])
                else:
                    n_next = n+self.n_args[i]
                    par = params[n:n_next] 
            else:
                n_next = n+self.n_args[i]
                par = params[n:n_next]            
            
            
            data += np.array(functions[i](x,*par))*params[-len(functions)+i] 
            n = n_next
        data /= np.sum(params[-len(functions):])
        return data
    


def BendingPL(v,A,v_bend,a_low,a_high,c):
    
    numer = v**-a_low
    denom = 1 + (v/v_bend)**(a_high-a_low)
    out = A * (numer/denom) + c
    return out

def RandAnyDist(f,args,a,b,size=1):
    
    
    out = []
    while len(out) < size:    
        x = rnd.rand()*(b-a) + a  # random value in x range
        v = f(x,*args)            # equivalent probability
        p = rnd.rand()      # random number
        if p <= v:
            out.append(x)   # add to value sample if random number < probability
    if size == 1:
        return out[0]
    else:
        return out

def PDF_Sample(lc):
    
    
    if lc.bins == None:
        lc.bins = OptBins(lc.flux)
    
    pdf = np.histogram(lc.flux,bins=lc.bins)
    chances = pdf[0]/float(sum(pdf[0]))
    nNumbers = len(lc.flux)
    
    sample = np.random.choice(len(chances), nNumbers, p=chances)
    
    sample = np.random.uniform(pdf[1][:-1][sample],pdf[1][1:][sample])

    return sample

#-------- PDF Fitting ---------------------------------------------------------

def Min_PDF(params,hist,model,force_scipy=False):
    

    # pdf(x,shape,loc,scale)

    # gamma - shape = kappa,loc = 0, scale = theta
    # lognorm - shape = sigma, loc = 0, scale = exp(mu)

    mids = (hist[1][:-1]+hist[1][1:])/2.0

    try:
        if model.__name__ == 'Mixture_Dist':
            model = model.Value
            m = model(mids,params)
        elif model.__module__ == 'scipy.stats.distributions' or \
            model.__module__ == 'scipy.stats._continuous_distns' or \
                force_scipy == True:
            m = model.pdf    
        else:    
            m = model(mids,*params)
    except AttributeError:
        m = model(mids,*params)
    
    chi = (hist[0] - m)**2.0
    
    return np.sum(chi)
 
def OptBins(data,maxM=100):
    
    
    N = len(data)
    
    # loop through the different numbers of bins
    # and compute the posterior probability for each.
    
    logp = np.zeros(maxM)
    
    for M in range(1,maxM+1):
        n = np.histogram(data,bins=M)[0] # Bin the data (equal width bins)
        
        # calculate posterior probability
        part1 = N * np.log(M) + sp.gammaln(M/2.0)
        part2 = - M * sp.gammaln(0.5)  - sp.gammaln(N + M/2.0)
        part3 = np.sum(sp.gammaln(n+0.5))
        logp[M-1] = part1 + part2 + part3 # add to array of posteriors

    maximum = np.argmax(logp) + 1 # find bin number of maximum probability
    return maximum + 10 
    

#--------- PSD fitting --------------------------------------------------------

def PSD_Prob(params,periodogram,model):
    psd = model(periodogram[0],*params) # calculate the psd

    even = True   
    
    # calculate the likelihoods for each value THESE LINES CAUSE RUNTIME ERRORS
    if even:
        p = 2.0 * np.sum( np.log(psd[:-1]) + (periodogram[1][:-1]/psd[:-1]) )
        p_nq = np.log(np.pi * periodogram[1][-1]*psd[-1]) \
                                         + 2.0 * (periodogram[1][-1]/psd[-1])
        p += p_nq
    else:
        p = 2.0 * np.sum( np.log(psd) + (periodogram[1]/psd) )
        
    return p


         
#--------- Standard Deviation estimate ----------------------------------------
 
def SD_estimate(mean,v_low,v_high,PSDdist,PSDdistArgs):
    
    i = itg.quad(PSDdist,v_low,v_high,tuple(PSDdistArgs))
    out = [np.sqrt(mean**2.0 * i[0]),np.sqrt(mean**2.0 * i[1])]
    return out

#------------------ Lightcurve Simulation Functions ---------------------------

def TimmerKoenig(RedNoiseL, aliasTbin, randomSeed, tbin, LClength,\
                    PSDmodel, PSDparams,std=1.0, mean=0.0):    
                        
    # --- create freq array up to the Nyquist freq & equivalent PSD ------------
    frequency = np.arange(1.0, (RedNoiseL*LClength)/2 +1)/ \
                                            (RedNoiseL*LClength*tbin*aliasTbin)
    powerlaw = PSDmodel(frequency,*PSDparams)

    # -------- Add complex Gaussian noise to PL --------------------------------
    rnd.seed(randomSeed)
    real = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,((RedNoiseL*LClength)/2))
    imag = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,((RedNoiseL*LClength)/2))
    positive = np.vectorize(complex)(real,imag) # array of +ve, complex nos
    noisypowerlaw = np.append(positive,positive.conjugate()[::-1])
    znoisypowerlaw = np.insert(noisypowerlaw,0,complex(0.0,0.0)) # add 0

    # --------- Fourier transform the noisy power law --------------------------
    inversefourier = np.fft.ifft(znoisypowerlaw)  # should be ONLY  real numbers
    longlightcurve = inversefourier.real       # take real part of the transform
 
    # extract random cut and normalise output lightcurve, 
    # produce fft & periodogram
    if RedNoiseL == 1:
        lightcurve = longlightcurve
    else:
        extract = rnd.randint(LClength-1,RedNoiseL*LClength - LClength)
        lightcurve = np.take(longlightcurve,list(range(extract,extract + LClength)))

    if mean: 
        lightcurve = lightcurve-np.mean(lightcurve)
    if std:
        lightcurve = (lightcurve/np.std(lightcurve))*std
    if mean:
        lightcurve += mean

    fft = ft.fft(lightcurve)

    periodogram = np.absolute(fft)**2.0 * ((2.0*tbin*aliasTbin*RedNoiseL)/\
                   (LClength*(np.mean(lightcurve)**2)))   
    shortPeriodogram = np.take(periodogram,list(range(1,LClength/2 +1)))
    #shortFreq = np.take(frequency,range(1,LClength/2 +1))
    shortFreq = np.arange(1.0, (LClength)/2 +1)/ (LClength*tbin)
    shortPeriodogram = [shortFreq,shortPeriodogram]

    return lightcurve, fft, shortPeriodogram

# The Emmanoulopoulos Loop

def EmmanLC(time,RedNoiseL,aliasTbin,RandomSeed,tbin,
              PSDmodel, PSDparams, PDFmodel=None, PDFparams=None,maxFlux=None,
                    maxIterations=1000,verbose=False, LClength=None, \
                        force_scipy=False,histSample=None):

    if LClength != None:
        length = LClength
        time = np.arange(0,tbin*LClength,tbin)
    else:
        length = len(time)

    ampAdj = None
    
    # Produce Timmer & Koenig simulated LC
    if verbose:
        print("Running Timmer & Koening...")
        
    tries = 0      
    success = False
    
    if histSample:
        mean = np.mean(histSample.mean)
    else:
        mean = 1.0
    
    while success == False and tries < 5:
        try:
            if LClength:
                shortLC, fft, periodogram = \
                    TimmerKoenig(RedNoiseL,aliasTbin,RandomSeed,tbin,LClength,
                                             PSDmodel,PSDparams,mean=1.0)
                success = True               
            else:
                shortLC, fft, periodogram = \
                    TimmerKoenig(RedNoiseL,aliasTbin,RandomSeed,tbin,len(time),
                                             PSDmodel,PSDparams,mean=1.0)
                success = True
        # This has been fixed and should never happen now in theory...
        except IndexError:
            tries += 1
            print("Simulation failed for some reason (IndexError) - restarting...")

    shortLC = [np.arange(len(shortLC))*tbin, shortLC]
    
    # Produce random distrubtion from PDF, up to max flux of data LC
    # use inverse transform sampling if a scipy dist
    if histSample:
        dist = PDF_Sample(histSample)
    else:
        mix = False
        scipy = False
        try:
            if PDFmodel.__name__ == "Mixture_Dist":
                mix = True
        except AttributeError:
            mix = False
        try:
            if PDFmodel.__module__ == 'scipy.stats.distributions' or \
                PDFmodel.__module__ == 'scipy.stats._continuous_distns' or \
                    force_scipy == True:
                scipy = True
        except AttributeError:
            scipy = False
    
        if mix:     
            if verbose: 
                print("Inverse tranform sampling...")
            dist = PDFmodel.Sample(PDFparams,length)
        elif scipy:
            if verbose: 
                print("Inverse tranform sampling...")
            dist = PDFmodel.rvs(*PDFparams,size=length)
            
        else: # else use rejection
            if verbose: 
                print("Rejection sampling... (slow!)")
            if maxFlux == None:
                maxFlux = 1
            dist = RandAnyDist(PDFmodel,PDFparams,0,max(maxFlux)*1.2,length)
            dist = np.array(dist)
        
    if verbose:
        print("mean:",np.mean(dist))
    sortdist = dist[np.argsort(dist)] # sort!
    
    # Iterate over the random sample until its PSD (and PDF) match the data
    if verbose:
        print("Iterating...")
    i = 0
    oldSurrogate = np.array([-1])
    surrogate = np.array([1])
    
    while i < maxIterations and np.array_equal(surrogate,oldSurrogate) == False:
    
        oldSurrogate = surrogate
    
        if i == 0:
            surrogate = [time, dist] # start with random distribution from PDF
        else:
            surrogate = [time,ampAdj]#
            
        ffti = ft.fft(surrogate[1])
        
        PSDlast = ((2.0*tbin)/(length*(mean**2))) *np.absolute(ffti)**2
        PSDlast = [periodogram[0],np.take(PSDlast,list(range(1,length/2 +1)))]
        
        fftAdj = np.absolute(fft)*(np.cos(np.angle(ffti)) \
                                    + 1j*np.sin(np.angle(ffti)))  #adjust fft
        LCadj = ft.ifft(fftAdj)
        LCadj = [time/tbin,LCadj]

        PSDLCAdj = ((2.0*tbin)/(length*np.mean(LCadj)**2.0)) \
                                                 * np.absolute(ft.fft(LCadj))**2
        PSDLCAdj = [periodogram[0],np.take(PSDLCAdj, list(range(1,length/2 +1)))]
        sortIndices = np.argsort(LCadj[1])
        sortPos = np.argsort(sortIndices)
        ampAdj = sortdist[sortPos]
        
        i += 1
    if verbose:
        print("Converged in {} iterations".format(i))
    
    return surrogate, PSDlast, shortLC, periodogram, ffti

#--------- Lightcurve Class & associated functions -----------------------------

class Lightcurve(object):
    
    def __init__(self,time,flux,tbin,errors=None):
         self.time = time
         self.flux  = flux
         self.errors = errors
         self.length = len(time)
         self.freq = np.arange(1, self.length/2.0 + 1)/(self.length*tbin)
         self.psd = None 
         self.mean = np.mean(flux)
         self.std_est = None
         self.std = np.std(flux) 
         self.tbin = tbin
         self.fft = None
         self.periodogram = None
         self.pdfFit = None
         self.psdFit = None
         self.psdModel = None
         self.pdfModel = None
         self.bins = None

    def Simulate_DE_Lightcurve(self,PSDmodel=None,PSDinitialParams=None,
                               PSD_fit_method='Nelder-Mead',n_iter=1000, 
                               PDFmodel=None,PDFinitialParams=None, 
                               PDF_fit_method = 'BFGS',nbins=None,
                               RedNoiseL=100, aliasTbin=1,randomSeed=None,
                                   maxIterations=1000,verbose=False,size=1,LClength=None,histSample=False):
                                       
        
        # fit PDF and PSD if necessary 
                           
        if self.psdFit == None:
            
            if PSDmodel:
                "Fitting PSD with supplied model..."
                self.Fit_PSD( initial_params= PSDinitialParams, 
                                 model = PSDmodel, fit_method=PSD_fit_method,
                                     n_iter=n_iter,verbose=False)
            else:
                print("PSD not fitted, fitting using defaults (bending power law)...")
                self.Fit_PSD(verbose=False)
        if self.pdfFit == None and histSample == False:
            if PDFmodel:
                "Fitting PDF with supplied model..."
                self.Fit_PDF(initial_params=PDFinitialParams,
                        model= PDFmodel, fit_method = PDF_fit_method,
                        verbose=False, nbins=nbins)
            else:
                print("PDF not fitted, fitting using defaults (gamma + lognorm)")
                self.Fit_PDF(verbose=False)    
        
        # check if fits were successful
        if self.psdFit == None or (self.pdfFit == None and histSample == False): 
            print("Simulation terminated due to failed fit(s)")
            return
            
        # check if standard deviation has been given
        self.STD_Estimate(self.psdModel,self.psdFit['x'])
        
        # simulate lightcurve
        if histSample:
            lc = Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                                size = size,\
                         LClength=LClength,lightcurve=self,histSample=True)
            
        else:
            lc = Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                self.pdfModel,self.pdfFit['x'], size = size,\
                         LClength=LClength,lightcurve=self,histSample=False)
         
        return lc

    def STD_Estimate(self,PSDdist=None,PSDdistArgs=None):
        
        if PSDdist == None:
            if self.psdModel == None:
                print("Fitting PSD for standard deviation estimation...")
                self.Fit_PSD(verbose=False)
                
            PSDdist = self.psdModel
            PSDdistArgs = self.psdFit['x']
        
        self.std_est = SD_estimate(self.mean,self.freq[0],self.freq[-1],PSDdist,PSDdistArgs)[0]
        return self.std_est  

    def PSD(self,PSDdist,PSDdistArgs):   
        
        self.psd = PSDdist(self.freq,*PSDdistArgs)
        return self.psd

    def Fourier_Transform(self):
        
        self.fft = ft.fft(self.flux) # 1D Fourier Transform (as time-binned)
        return self.fft
    
    def Periodogram(self):
        
        if self.fft == None:
            self.Fourier_Transform()
        periodogram = ((2.0*self.tbin)/(self.length*(self.mean**2)))\
                            * np.absolute(np.real(self.fft))**2
        freq = np.arange(1, self.length/2 + 1).astype(float)/\
                                                         (self.length*self.tbin)
        shortPeriodogram = np.take(periodogram,np.arange(1,self.length/2 +1))
        self.periodogram = [freq,shortPeriodogram]

        return self.periodogram
    
    def Set_PSD_Fit(self, model, params):
        self.psdModel = model
        self.psdFit = dict([('x',np.array(params))])
    
    def Set_PDF_Fit(self, model, params=None):
        if model == None:
            self.pdfModel = None
            self.pdfFit = None
        else:
            self.pdfModel = model
            if params:
                self.pdfFit = dict([('x',np.array(params))])
        
    def Fit_PSD(self, initial_params= [1, 0.001, 1.5, 2.5, 0], 
                model = BendingPL, fit_method='Nelder-Mead',n_iter=1000,
                verbose=True):
        
        if self.periodogram == None:
            self.Periodogram()
          
        minimizer_kwargs = {"args":(self.periodogram,model),"method":fit_method}
        m = op.basinhopping(PSD_Prob, initial_params,
                                 minimizer_kwargs=minimizer_kwargs,niter=n_iter)
         
        if m['message'][0] == \
           'requested number of basinhopping iterations completed successfully':
            
            if verbose:
                print("\n### Fit successful: ###")
                #A,v_bend,a_low,a_high,c
                
                if model == BendingPL:
                    print("Normalisation: {}".format(m['x'][0]))
                    print("Bend frequency: {}".format(m['x'][1]))
                    print("Low frequency index: {}".format(m['x'][2]))
                    print("high frequency index: {}".format(m['x'][3]))
                    print("Offset: {}".format(m['x'][4]))

                else:
                    for p in range(len(m['x'])):
                        print("Parameter {}: {}".format(p+1,m['x'][p]))
            
            self.psdFit = m 
            self.psdModel = model

        else:
            print("#### FIT FAILED ####")
            print("Fit parameters not assigned.")
            print("Try different initial parameters with the 'initial_params' keyword argument") 
  
    
    def Fit_PDF(self, initial_params=[6,6, 0.3,7.4,0.8,0.2],
                     model= None, fit_method = 'BFGS', nbins=None,verbose=True):
        
        
        if model == None:
            model = Mixture_Dist([st.gamma,st.lognorm],
                                            [3,3],[[[2],[0]],[[2],[0],]])
            model.default = True
        
        if nbins == None:
            nbins = OptBins(self.flux)        
        self.bins = nbins
        hist = np.array(np.histogram(self.flux,bins=nbins,normed=True))
        
        m = op.minimize(Min_PDF, initial_params, 
                            args=(hist,model),method=fit_method)
        
        if m['success'] == True:
            
            if verbose:
                print("\n### Fit successful: ###")

                mix = False
                try:
                    if model.__name__ == "Mixture_Dist":
                        mix = True
                except AttributeError:
                    mix = False

                if mix:
                    if model.default == True:                        
                        #kappa,theta,lnmu,lnsig,weight
                        print("\nGamma Function:")
                        print("kappa: {}".format(m['x'][0]))
                        print("theta: {}".format(m['x'][1]))
                        print("weight: {}".format(m['x'][4]))
                        print("\nLognormal Function:")
                        print("exp(ln(mu)): {}".format(m['x'][2]))
                        print("ln(sigma): {}".format(m['x'][3]))
                        print("weight: {}".format(1.0 - m['x'][4]))
                    else:
                        for p in range(len(m['x'])):
                            print("Parameter {}: {}".format(p+1,m['x'][p]))            
                else:
                    for p in range(len(m['x'])):
                        print("Parameter {}: {}".format(p+1,m['x'][p]))
                        
            self.pdfFit = m 
            self.pdfModel = model

        else:
            print("#### FIT FAILED ####")
            print("Fit parameters not assigned.")
            print("Try different initial parameters with the 'initial_params' keyword argument") 
              
    
    def Simulate_DE_Lightcurve(self,PSDmodel=None,PSDinitialParams=None,
                               PSD_fit_method='Nelder-Mead',n_iter=1000, 
                               PDFmodel=None,PDFinitialParams=None, 
                               PDF_fit_method = 'BFGS',nbins=None,
                               RedNoiseL=100, aliasTbin=1,randomSeed=None,
                                   maxIterations=1000,verbose=False,size=1,LClength=None,histSample=False):                           
        
        # fit PDF and PSD if necessary 
                           
        if self.psdFit == None:
            
            if PSDmodel:
                "Fitting PSD with supplied model..."
                self.Fit_PSD( initial_params= PSDinitialParams, 
                                 model = PSDmodel, fit_method=PSD_fit_method,
                                     n_iter=n_iter,verbose=False)
            else:
                print("PSD not fitted, fitting using defaults (bending power law)...")
                self.Fit_PSD(verbose=False)
        if self.pdfFit == None and histSample == False:
            if PDFmodel:
                "Fitting PDF with supplied model..."
                self.Fit_PDF(initial_params=PDFinitialParams,
                        model= PDFmodel, fit_method = PDF_fit_method,
                        verbose=False, nbins=nbins)
            else:
                print("PDF not fitted, fitting using defaults (gamma + lognorm)")
                self.Fit_PDF(verbose=False)    
        
        # check if fits were successful
        if self.psdFit == None or (self.pdfFit == None and histSample == False): 
            print("Simulation terminated due to failed fit(s)")
            return
            
        # check if standard deviation has been given
        self.STD_Estimate(self.psdModel,self.psdFit['x'])
        
        # simulate lightcurve
        if histSample:
            lc = Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                                size = size,\
                         LClength=LClength,lightcurve=self,histSample=True)
            
        else:
            lc = Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                self.pdfModel,self.pdfFit['x'], size = size,\
                         LClength=LClength,lightcurve=self,histSample=False)
         
        return lc
                                   
    def Plot_Periodogram(self):
        
        if self.periodogram == None:
            self.Periodogram()

        p = plt.subplot(1,1,1)
        plt.scatter(self.periodogram[0],self.periodogram[1])
        
        if self.psdFit:
            plx = np.logspace(np.log10(0.8*min(self.periodogram[0])),np.log10(1.2*max(self.periodogram[0])),100)          
            mpl = self.psdModel(plx,*self.psdFit['x'])
            plt.plot(plx,mpl,color='red',linewidth=3)
            
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')

        plt.xlim([0.8*min(self.periodogram[0]),1.2*max(self.periodogram[0])])
        plt.ylim([0.8*min(self.periodogram[1]),1.2*max(self.periodogram[1])])
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)
        plt.show()

    def Plot_Lightcurve(self):
        '''
        Plot the lightcurve.
        '''
        #plt.errorbar(self.time,self.flux,yerr=self.errors,)
        plt.scatter(self.time,self.flux)
        plt.title("Lightcurve")
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)
        plt.show()
        
    def Plot_PDF(self,bins=None,norm=True,force_scipy=False):
        '''
        Plot the probability density function of the lightcurve, and the model
        fit if present.
        '''

        if self.bins == None:
            self.bins = OptBins(self.flux)
        plt.hist(self.flux,bins=self.bins,normed=norm)
        #kappa,theta,lnmu,lnsig,weight
        if self.pdfFit:
            x = np.arange(0,np.max(self.flux)*1.2,0.01)
            
            mix = False
            scipy = False
            try:
                if self.pdfModel.__name__ == "Mixture_Dist":
                    mix = True
            except AttributeError:
                mix = False
            try:
                if self.pdfModel.__module__ == 'scipy.stats.distributions' or \
                    self.pdfModel.__module__ == 'scipy.stats._continuous_distns' or \
                        force_scipy == True:
                    scipy = True
            except AttributeError:
                scipy = False

            if mix:
                plt.plot(x,self.pdfModel.Value(x,self.pdfFit['x']),
                       'red', linewidth=3) 
            elif scipy:
                plt.plot(x,self.pdfModel.pdf(x,*self.pdfFit['x']),
                       'red', linewidth=3) 
            else:
                plt.plot(x,self.pdfModel(x,*self.pdfFit['x']),
                       'red', linewidth=3)  

        plt.title("PDF")
        plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)  
        plt.show()
                
    def Plot_Stats(self,bins=None,norm=True,force_scipy=False):
        '''
        Plot the lightcurve together with its probability density function and
        power spectral density.
        '''
                    
        if self.periodogram == None:
            self.Periodogram()
        plt.figure()
        plt.subplot(3,1,1)
        plt.scatter(self.time,self.flux)
        plt.title("Lightcurve")
        plt.subplot(3,1,2)
        if bins:
            plt.hist(self.flux,bins=bins,normed=norm)        
        else:
            if self.bins == None:
                self.bins = OptBins(self.flux)
            plt.hist(self.flux,bins=self.bins,normed=norm)        
        if self.pdfFit:
            x = np.arange(0,np.max(self.flux)*1.2,0.01)

            mix = False
            scipy = False
            try:
                if self.pdfModel.__name__ == "Mixture_Dist":
                    mix = True
            except AttributeError:
                mix = False
            try:
                if self.pdfModel.__module__ == 'scipy.stats.distributions' or \
                    self.pdfModel.__module__ == 'scipy.stats._continuous_distns' or \
                        force_scipy == True:
                    scipy = True
            except AttributeError:
                scipy = False

            if mix:
                plt.plot(x,self.pdfModel.Value(x,self.pdfFit['x']),
                       'red', linewidth=3) 
            elif scipy:
                plt.plot(x,self.pdfModel.pdf(x,*self.pdfFit['x']),
                       'red', linewidth=3) 
            else:
                plt.plot(x,self.pdfModel(x,*self.pdfFit['x']),
                       'red', linewidth=3) 
        plt.title("PDF")
        p=plt.subplot(3,1,3)
        plt.scatter(self.periodogram[0],self.periodogram[1])
        if self.psdFit:
            plx = np.logspace(np.log10(0.8*min(self.periodogram[0])),np.log10(1.2*max(self.periodogram[0])),100)            
            #print self.psdFit['x']
            mpl = self.psdModel(plx,*self.psdFit['x'])
            plt.plot(plx,mpl,color='red',linewidth=3)  
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')
        plt.xlim([0.8*min(self.periodogram[0]),1.2*max(self.periodogram[0])])
        plt.ylim([0.8*min(self.periodogram[1]),1.2*max(self.periodogram[1])])
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3) 
        plt.show()
        
    def Save_Periodogram(self,filename,dataformat='%.10e'):
        
        if self.periodogram == None:
            self.Periodogram()        
        
        data = np.array([self.periodogram[0],self.periodogram[1]]).T
        np.savetxt(filename,data,fmt=dataformat,header="Frequency\tPower")
        
    def Save_Lightcurve(self,filename,dataformat='%.10e'):
        
        
        data = np.array([self.time,self.flux]).T
        np.savetxt(filename,data,fmt=dataformat,header="Time\tFlux")

def Comparison_Plots(lightcurves,bins=None,norm=True, names=None,
                overplot=False,colors=['blue','red','green','black','orange'],\
                    force_scipy=False):
    
   
    if overplot:
        n = 1
    else:
        n = len(lightcurves)
        
    if names == None or len(names) != n:
        names = ["Lightcurve {}".format(l) for l in range(1,n+1)]
    i = 0
    c = 0

    plt.figure()
    for lc in lightcurves:
                

        # calculate periodogram if not yet done
        if lc.periodogram == None:
            lc.Periodogram()
           
        # lightcurve
        plt.subplot(3,n,1+i)
        plt.scatter(lc.time,lc.flux,color=colors[c])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title(names[i])
        
        # PDF
        plt.subplot(3,n,n+1+i)
        
        # calculate/get optimum bins if not given
        if bins == None:
            if lc.bins == None:
                lc.bins = OptBins(lc.flux)
            bins = lc.bins
        plt.hist(lc.flux,bins=bins,normed=norm,color=colors[c] )
        if lightcurves[-1].pdfFit:
            x = np.arange(0,np.max(lightcurves[-1].flux)*1.2,0.01)

            mix = False
            scipy = False
            try:
                if lightcurves[-1].pdfModel.__name__ == "Mixture_Dist":
                    mix = True
            except AttributeError:
                mix = False
            try:
                if lightcurves[-1].pdfmodel.__module__ == \
                                                'scipy.stats.distributions' or \
                    lightcurves[-1].pdfmodel.__module__ == \
                                           'scipy.stats._continuous_distns' or \
                        force_scipy == True:
                    scipy = True
            except AttributeError:
                scipy = False

            if mix:
                plt.plot(x,lightcurves[-1].pdfModel.Value(x,lightcurves[-1].pdfFit['x']),
                       'red', linewidth=3) 
            elif scipy:
                plt.plot(x,lightcurves[-1].pdfModel.pdf(x,lightcurves[-1].pdfFit['x']),
                       'red', linewidth=3) 
            else:
                plt.plot(x,lightcurves[-1].pdfModel(x,*lightcurves[-1].pdfFit['x']),
                       'red', linewidth=3 ) 
        plt.title("PDF")
        
        # PSD
        p=plt.subplot(3,n,2*n+1+i)
        plt.scatter(lc.periodogram[0],lc.periodogram[1],color=colors[c])
        if lightcurves[-1].psdFit:
            plx = np.logspace(np.log10(0.8*min(lc.periodogram[0])),np.log10(1.2*max(lc.periodogram[0])),100)          
            mpl = lc.psdModel(plx,*lightcurves[-1].psdFit['x'])
            plt.plot(plx,mpl,color='red',linewidth=3) 
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')
        plt.xlim([0.8*min(lc.periodogram[0]),1.2*max(lc.periodogram[0])])
        plt.ylim([0.8*min(lc.periodogram[1]),1.2*max(lc.periodogram[1])])

        if overplot:
            c += 1
        else:
            i += 1
            
    # adjust plots
    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)      
    plt.show()    
        
def Load_Lightcurve(fileroute, tbin,
                    time_col=None, flux_col=None, error_col=None,
                    extension=1, element=None):
    two = False    
    print(f"\nLoading lightcurve from {fileroute}...")

    if time_col is not None and flux_col is not None:
        print("Using FITS loader (with columns)")
        ...
    elif time_col or flux_col:
        print("*** LOAD FAILED ***")
        print("Both time_col and flux_col must be defined for FITS input.")
        return
    else:
        print("Using plain text loader...")
        try:
            time, flux, error = [], [], []
            success = False
            read = 0

            with open(fileroute, 'r') as file:
                for i, line in enumerate(file):
                    line = line.strip()
                    if ',' in line:
                        columns = line.split(',')
                    else:
                        columns = line.split()

                    if i == 0 and not columns[0].replace('.', '', 1).isdigit():
                         print("Skipping header:", columns)
                         continue

                    print(f"Line {i}: {columns}")  # Debug line

                    if len(columns) == 3:
                        t_val, f_val, e_val = columns
                        try:
                            time.append(float(t_val))
                            flux.append(float(f_val))
                            error.append(float(e_val))
                            read += 1
                            success = True
                        except ValueError:
                            print(f"ValueError on line {i} - skipping")
                            continue
                    elif len(columns) == 2:
                        t_val, f_val = columns
                        try:
                            time.append(float(t_val))
                            flux.append(float(f_val))
                            read += 1
                            success = True
                            two = True
                        except ValueError:
                            print(f"ValueError on line {i} - skipping")
                            continue

            if success:
                print(f"✅ Successfully read {read} lines of data")
            else:
                print("*** LOAD FAILED ***")
                print("Text file must have 2 or 3 numeric columns")
                return

        except IOError:
            print("*** LOAD FAILED ***")
            print("Input file not found")
            return

    print("Creating Lightcurve object...")
    if two:
        return Lightcurve(np.array(time), np.array(flux), tbin)
    else:
        return Lightcurve(np.array(time), np.array(flux), tbin, np.array(error))
    return lc

def Simulate_TK_Lightcurve(PSDmodel,PSDparams,lightcurve=None,
                           tbin = None, length = None, mean = 0.0, std = 1.0,
                           RedNoiseL=100, aliasTbin=1,randomSeed=None):
    
   
    if type(lightcurve) == Lightcurve:
        if tbin == None:
            tbin = lightcurve.tbin
        if length == None:
            length = lightcurve.length
        if mean == None:
            mean  = lightcurve.mean
        if std == None:
            if lightcurve.std_est == None:
                std = lightcurve.std
            else:
                std = lightcurve.std    

        time = lightcurve.time
        
    else:
        time = np.arange(0,length*tbin)    
        
        if tbin == None:
            tbin = 1
        if length == None:
            length = 1000
        if mean == None:
            mean  = 1
        if std == None:
            std = 1
            
    shortLC, fft, periodogram = \
        TimmerKoenig(RedNoiseL,aliasTbin,randomSeed,tbin,
                     length, PSDmodel,PSDparams,std,mean)
    lc = Lightcurve(time,shortLC,tbin=tbin)
        
    lc.fft = fft
    lc.periodogram = periodogram

    lc.psdModel =   PSDmodel
    lc.psdFit = {'x':PSDparams}

    return lc

def Simulate_DE_Lightcurve(PSDmodel,PSDparams,PDFmodel=None, PDFparams=None,
                             lightcurve = None, tbin = None, LClength = None, 
                               maxFlux = None,
                                 RedNoiseL=100, aliasTbin=1,randomSeed=None,
                                   maxIterations=1000,verbose=False,size=1,histSample=False):
    

    lcs = np.array([])

    if type(lightcurve) == Lightcurve:
        if tbin == None:
            tbin = lightcurve.tbin
        #### NOT NEEDED #####
        #if LClength == None:
        #    LClength = lightcurve.length
        #if mean == None:
        #mean  = lightcurve.mean
        #if std == None:
        #if lightcurve.std_est == None:
        #    std = lightcurve.std
        #else:
        #    std = lightcurve.std    

        time = lightcurve.time
        
        
        if maxFlux == None:
            maxFlux = max(lightcurve.flux)
        
    else: 
        if tbin == None:
            tbin = 1
        if LClength == None:
            LClength = 100
        #### NOT NEEDED #####
        #if mean == None:
        #    mean  = 1
        #if std == None:
        #    std = 1    

        time = np.arange(0,LClength*tbin)             
    
    for n in range(size): 
                   
        surrogate, PSDlast, shortLC, periodogram, fft = \
            EmmanLC(time, RedNoiseL,aliasTbin,randomSeed, tbin,
                            PSDmodel, PSDparams, PDFmodel, PDFparams, maxFlux,
                                maxIterations,verbose,LClength,histSample=lightcurve)
        lc = Lightcurve(surrogate[0],surrogate[1],tbin=tbin)
        lc.fft = fft
        lc.periodogram = PSDlast
        
        lc.psdModel =   PSDmodel
        lc.psdFit = {'x':PSDparams}
        if histSample == False:
            lc.pdfModel = PDFmodel
            lc.pdfFit = {'x':PDFparams}
        
        lcs = np.append(lcs,lc)
        
    if n > 1:
        return lcs
    else:
        return lc


