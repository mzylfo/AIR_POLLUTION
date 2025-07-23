
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os

class NeuroDistributions():

    def __init__(self, path_folder, data, univ_id):
        # Load data from statsmodels datasets
        self.data = pd.Series(data)
        self.path_folder = Path(path_folder,"img_dist")
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.univ_id = univ_id
 

    # Create models from data
    def best_fit_distribution(self, bins=200, ax=None):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(self.data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_distributions = []

        # Estimate distribution parameters from data
        for ii, distribution_name in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

            #print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution_name))

            distribution = getattr(st, distribution_name)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # fit dist to data
                    params = distribution.fit(self.data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax, label=distribution_name)
                        
                    except Exception:
                        pass

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))
            
            except Exception:
                pass

        
        return sorted(best_distributions, key=lambda x:x[2])

    def make_pdf(self, dist, params, size=10000):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    def plotDistributions(self, save_plot=True):

        matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
        matplotlib.style.use('ggplot')

        # Plot for comparison
        plt.figure(figsize=(12,8))
        ax = self.data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

        # Save plot limits
        dataYLim = ax.get_ylim()

        # Find best fit distribution
        best_distibutions = self.best_fit_distribution(200, ax)
        best_dist = best_distibutions[0]

        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(u'All distributions comparison')
        
        #plt.legend(loc="upper left")
        if save_plot:
            filename = Path(self.path_folder,"dist_"+str(self.univ_id)+"_all.png")
            plt.savefig(filename)
            


        # Make PDF with best params 
        pdf = self.make_pdf(best_dist[0], best_dist[1])
        

        # Display
        plt.figure(figsize=(12,8))
        ax = pdf.plot(lw=2, label=f'PDF - {best_dist[0].name}', legend=True)
        self.data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

        param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
        dist_str = '{}({})'.format(best_dist[0].name, param_str)

        ax.set_title(u'With best fit distribution \n' + dist_str)
        if save_plot:
            filename = Path(self.path_folder,"dist_"+str(self.univ_id)+"_best.png")
            plt.savefig(filename)
        
    def get_best_dist(self):
        best_distibutions = self.best_fit_distribution(200, None)
        return best_distibutions[0]
