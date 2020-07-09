import numpy.random as npr
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.stats import expon, lognorm, norm, gamma
from scipy.stats._distn_infrastructure import rv_frozen


def make_normal_from_interval(lb, ub, alpha):
    ''' Creates a normal distribution SciPy object from intervals.

    This function is a helper to create SciPy distributions by specifying the
    amount of wanted density between a lower and upper bound. For example,
    calling with (lb, ub, alpha) = (2, 3, 0.95) will create a Normal
    distribution with 95% density between 2 a 3.

    Args:
        lb (float): Lower bound
        ub (float): Upper bound
        alpha (float): Total density between lb and ub

    Returns:
        scipy.stats.norm
    
    Examples:
        >>> dist = make_normal_from_interval(-1, 1, 0.63)
        >>> dist.mean()
        0.0
        >>> dist.std()
        1.1154821104064199
        >>> dist.interval(0.63)
        (-1.0000000000000002, 1.0)

    '''
    z = norm().interval(alpha)[1]
    mean_norm = (ub + lb) / 2
    std_norm = (ub - lb) / (2 * z)
    return norm(loc=mean_norm, scale=std_norm)


def make_lognormal_from_interval(lb, ub, alpha):
    ''' Creates a lognormal distribution SciPy object from intervals.

    This function is a helper to create SciPy distributions by specifying the
    amount of wanted density between a lower and upper bound. For example,
    calling with (lb, ub, alpha) = (2, 3, 0.95) will create a LogNormal
    distribution with 95% density between 2 a 3.

    Args:
        lb (float): Lower bound
        ub (float): Upper bound
        alpha (float): Total density between lb and ub

    Returns:
        scipy.stats.lognorm
    
    Examples:
        >>> dist = make_lognormal_from_interval(2, 3, 0.95)
        >>> dist.mean()
        2.46262863041182
        >>> dist.std()
        0.25540947842844575
        >>> dist.interval(0.95)
        (1.9999999999999998, 2.9999999999999996)

    '''
    z = norm().interval(alpha)[1]
    mean_norm = np.sqrt(ub * lb)
    std_norm = np.log(ub / lb) / (2 * z)
    return lognorm(s=std_norm, scale=mean_norm)


class EmpiricalDistribution:
    def __init__(self, observations, method='sequential'):
        self.observations = np.array(observations)
        self.method = 'sequential'
        self.rvs = (self._sequential_rvs if method == 'sequential' else
                    self._uniform_rvs)

    def _sequential_rvs(self, size):
        assert size <= len(self.observations)
        return self.observations[:size]

    def _uniform_rvs(self, size):
        return np.random.choice(self.observations, size, replace=True)


class SEIRBayes:
    ''' Model with Susceptible, Exposed, Infectious and Recovered compartments.

    This class implements the SEIR model with stochastic incubation and
    infectious periods as well as the basic reproduction number R0. 

    This model is an implicit density function on 4 time series S(t), E(t), 
    I(t) and R(t) for t = 0 to t_max-1. Sampling is done via numerical 
    resolution of a system of stochastic differential equations with 6 
    degrees of randomness: alpha, gamma, r0 and the number of subjects 
    transitioning between compartments; S -> E, E -> I, I -> R.

    Infectious (1/gamma) and incubation (1/alpha) periods, as well as basic 
    reproduction number r0, can be specified in 3 ways: 
        * 4-tuple as (lower bound, upper bound, density value, dist. family);
        * SciPy distribution objects, taken from scipy.stats;
        * array-like containers such as lists, numpy arrays and pandas Series.
    Se the __init__ method for greater detail.

    The probability of an individual staying in a compartment up to time t
    is proportional to exp(-p*t), therefore the probability of leaving is
    1 - exp(-p*t). The rate p is different for each pair of source and 
    destination compartments. They are as follows

        ======== ============= ========== ============== 
         Source   Destination     Rate        Period     
        ======== ============= ========== ============== 
         S        E             beta*I/N   1/(beta*I/N)  
         E        I             alpha      1/alpha       
         I        R             gamma      1/gamma       
        ======== ============= ========== ============== 

    Since the above discussion is for a single individual, a Binomial 
    distribution with success rate 1 - exp(-p*t) is used to scale to the
    total number of individuals in each compartment.

    Attributes:
        params (dict): Summary of model parameter values. Is compatible with
            the constructor; SEIRBayes(*params).
        _params (dict): Similar to params, but with modifications to
            facilitate usage internally. Isn't compatible with the constructor.

    Examples:
        Default init.
        
        >>> np.random.seed(0)
        >>> model = SEIRBayes(t_max=5)
        >>> S, E, I, R, t_space = model.sample(3)
        >>> I
        array([[10., 10., 10.],
               [15., 10.,  9.],
               [17., 15.,  8.],
               [22., 17., 12.],
               [24., 18., 15.]])
        >>> model.params['r0_dist'].interval(0.95)
        (2.5, 6.0)


    '''
    def __init__(self, 
                 NEIR0=(100, 20, 10, 0),
                 r0_dist=(2.5, 6.0, 0.95, 'lognorm'),
                 gamma_inv_dist=(7, 14, 0.95, 'lognorm'),
                 alpha_inv_dist=(4.1, 7, 0.95, 'lognorm'),
                 fator_subr=1,
                 t_max=30):
        '''Default constructor method.


        Args:
            NEIR0 (tuple): Initial conditions in the form of 
                (population size, exposed, infected, recovered). Notice that
                S0, the initial susceptible population, is not needed as it 
                can be calculated as S0 = N - fator_subr*(E0 + I0 + R0).
            fator_subr (float): Multiplicative factor of I0 and E0 to take
                into account sub-reporting.
            t_max (int): Length of the time-series.

            r0_dist, alpha_inv_dist, and gamma_inv_dist can be specified as
            a tuple, scipy distribution, or array-like object.
                tuple: (lower bound, upper bound, density, dist. family)
                scipy dist: object from scipy.stats with rvs method
                array-like: the i-th value will be used for the i-th sample

            r0_dist (object): basic reproduction number.
            alpha_inv_dist (object): incubation period.
            gamma_inv_dist (object): infectious period.

        Examples:
            >>> np.random.seed(0)
            >>> model = SEIRBayes(fator_subr=2)
            >>> model.params['fator_subr']
            2
            >>> model.params['r0_dist'].rvs(10)
            array([5.74313347, 4.23505111, 4.81923138, 6.3885136 , 5.87744241,
                   3.11354468, 4.7884938 , 3.74424985, 3.78472191, 4.24493851])
            >>> model.params['NEIR0']
            (100, 20, 10, 0)
            >>> model._params['init_conditions']
            (40, 40, 20, 0)
            >>> model._params['total_population']
            100

            >>> np.random.seed(0)
            >>> model = SEIRBayes(r0_dist=(1.96, 5.1, 0.99, 'lognorm'),
            ...                   alpha_inv_dist=[5.1, 4.9, 6.0])
            >>> model.params['r0_dist'].rvs(10)
            array([4.38658658, 3.40543675, 3.79154822, 4.79256981, 4.4716837 ,
                   2.63710471, 3.7714376 , 3.07405105, 3.10164345, 3.41204358])
            >>> model.params['alpha_inv_dist'].rvs(3)
            array([5.1, 4.9, 6. ])
            
        '''
        r0_dist = self.init_param_dist(r0_dist)
        alpha_inv_dist = self.init_param_dist(alpha_inv_dist)
        gamma_inv_dist = self.init_param_dist(gamma_inv_dist)

        self.params = {
            'NEIR0': NEIR0,
            'r0_dist': r0_dist,
            'gamma_inv_dist': gamma_inv_dist,
            'alpha_inv_dist': alpha_inv_dist,
            'fator_subr': fator_subr,
            't_max': t_max
        }

        N, E0, I0, R0 = NEIR0
        S0 = N - fator_subr*(I0 + E0 + R0)

        self._params = {
            'init_conditions': (S0, fator_subr*E0, fator_subr*I0, fator_subr*R0),
            'fator_subr': fator_subr,
            'total_population': N,
            'alpha_inv_dist': alpha_inv_dist,
            'gamma_inv_dist': gamma_inv_dist,
            'r0_dist': r0_dist,
            't_max': t_max,
            'param_samples': {}
        }

    @classmethod
    def init_param_dist(cls, param_init):
        '''Initialize distribution from tuple, scipy or array-like object.

        Args:
            param_init (tuple, scipy.stats dist., or array-like)

        Examples:
            >>> np.random.seed(0)
            >>> dist = SEIRBayes.init_param_dist((1, 2, .9, 'lognorm'))
            >>> dist.interval(0.9)
            (1.0, 2.0)

            >>> dist = SEIRBayes.init_param_dist([0.1, 0.2, 0.3])
            >>> dist.rvs(2)
            array([0.1, 0.2])

            >>> from scipy.stats import lognorm
            >>> dist = SEIRBayes.init_param_dist(lognorm(s=.1, scale=1))
            >>> dist.mean()
            1.005012520859401

        '''
        if isinstance(param_init, tuple):
            lb, ub, density, family = param_init
            if family != 'lognorm':
                raise NotImplementedError('Only family lognorm '
                                          'is implemented')
            dist = make_lognormal_from_interval(lb, ub, density)
        elif isinstance(param_init, rv_frozen):
            dist = param_init
        else:
            dist = EmpiricalDistribution(param_init)
        return dist


    def sample(self, size=1, return_param_samples=False):
        '''Sample from model.
        Args:
            size (int): Number of samples.
            return_param_samples (bool): If true, returns the parameter
                samples (taken from {r0,gamma,alpha}_dist) used.

        Examples:

            >>> np.random.seed(0)
            >>> model = SEIRBayes(t_max=5)
            >>> S, E, I, R, t_space = model.sample(3)
            >>> S.shape, E.shape, I.shape, R.shape
            ((5, 3), (5, 3), (5, 3), (5, 3))
            >>> I
            array([[10., 10., 10.],
                   [15., 10.,  9.],
                   [17., 15.,  8.],
                   [22., 17., 12.],
                   [24., 18., 15.]])
            >>> t_space
            array([0, 1, 2, 3, 4])

            Return parameter samples for analysis.

            >>> np.random.seed(0)
            >>> model = SEIRBayes(t_max=5)
            >>> (S, E, I, R, t_space, r0,
            ...  alpha, gamma, beta) = model.sample(5, True)
            >>> r0
            array([5.74313347, 4.23505111, 4.81923138, 6.3885136 , 5.87744241])
            >>> alpha
            array([0.18303002, 0.15306351, 0.16825044, 0.18358956, 0.17569263])
            >>> gamma
            array([0.12007063, 0.08539356, 0.10375533, 0.1028759 , 0.09394099])
            >>> np.isclose(r0, beta/gamma).all()
            True
            >>> t_space
            array([0, 1, 2, 3, 4])
            
        '''
        t_space = np.arange(0, self._params['t_max'])
        N = self._params['total_population']
        S, E, I, R = [np.zeros((self._params['t_max'], size))
                      for _ in range(4)]
        S[0, ], E[0, ], I[0, ], R[0, ] = self._params['init_conditions']

        r0 = self._params['r0_dist'].rvs(size)
        gamma = 1/self._params['gamma_inv_dist'].rvs(size)
        alpha = 1/self._params['alpha_inv_dist'].rvs(size)
        beta = r0*gamma

        for t in t_space[1:]:
            SE = npr.binomial(S[t-1, ].astype(int),
                              expon(scale=1/(beta*I[t-1, ]/N)).cdf(1))
            EI = npr.binomial(E[t-1, ].astype(int),
                              expon(scale=1/alpha).cdf(1))
            IR = npr.binomial(I[t-1, ].astype(int),
                              expon(scale=1/gamma).cdf(1))

            dS =  0 - SE
            dE = SE - EI
            dI = EI - IR
            dR = IR - 0

            S[t, ] = S[t-1, ] + dS
            E[t, ] = E[t-1, ] + dE
            I[t, ] = I[t-1, ] + dI
            R[t, ] = R[t-1, ] + dR

        if return_param_samples:
            return S, E, I, R, t_space, r0, alpha, gamma, beta
        else:
            return S, E, I, R, t_space

class ReproductionNumber:

    def __init__(self, incidence, prior_shape=1, prior_scale=5,
                 si_pmf=None, si_pars=None, t_start=None, window_width=None):
        """
        Initialize ReproductionNumber class

        :param incidence: pandas DataFrame with columns 'dates' and 'incidence' (number of new cases per day).
        :param prior_shape: value of shape parameter of Gamma prior for reproduction number estimation.
        :param prior_scale: value of scale parameter of Gamma prior for reproduction number estimation.
        :param si_pmf: pandas DataFrame with columns 'interval_length' and 'probability'.
        Represents probability mass function for given values of serial interval.
        :param si_pars: dictionary with keys 'mean' and 'sd'.
        Represents parameters to generate PMF for serial interval.

        """

        self.incidence = incidence.reset_index().set_index('dates')
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.si_pmf = si_pmf
        self.si_pars = si_pars
        self.t_start = t_start
        self.window_width = window_width
        self.t_end = None
        self.posterior_parameters = {}
        self.posterior_summary = None
        self.check_time_periods()
        self.check_serial_number_pmf()

    def check_time_periods(self):
        if self.window_width is None:
            self.window_width = 6
        if self.t_start is None:
            self.t_start = np.arange(1, self.incidence.shape[0] - self.window_width)
        elif isinstance(self.t_start, list):
            self.t_start = np.array(self.t_start)
        self.t_end = self.t_start + self.window_width

    def check_serial_number_pmf(self):
        if self.si_pmf is not None and self.si_pars is not None:
            txt = "You must pass either 'si_pmf' or 'si_pars', not both."
            raise AttributeError(txt)
        if self.si_pmf is None:
            if self.si_pars is None:
                txt = "You must pass either 'si_pmf' or 'si_pars'. You've passed neither."
                raise AttributeError(txt)
            if not all([i in self.si_pars.keys() for i in ['mean', 'sd']]):
                txt = "'si_pars' must be a dictionary with 'mean' and 'sd' keys."
                raise AttributeError(txt)
            self.compute_serial_interval_pmf()
        else:
            self.si_pmf = self.si_pmf.reset_index().set_index('interval_length')['probability']

    def compute_serial_interval_pmf(self, k=None, mu=None, sigma=None):

        if k is None:
            k = np.arange(self.incidence.shape[0])
        elif not isinstance(k, np.ndarray):
            raise TypeError("k must be of type numpy.ndarray, probably shape = (n_time_windows, ).")

        if mu is None:
            mu = self.si_pars['mean']
        if sigma is None:
            sigma = self.si_pars['sd']

        if sigma < 0:
            raise AttributeError("sigma must be >=0.")
        if mu <= 1:
            raise AttributeError("mu must be >1")
        if not (k >= 0.).sum() == len(k):
            raise AttributeError("all values in k must be >=0.")

        shape = ((mu - 1) / sigma) ** 2
        scale = (sigma ** 2) / (mu - 1)

        def cdf_gamma(x, shape_, scale_):
            return gamma.cdf(x=x, a=shape_, scale=scale_)

        si_pmf = k * cdf_gamma(k,
                               shape,
                               scale) + (k - 2) * cdf_gamma(k - 2,
                                                            shape,
                                                            scale) - 2 * (k - 1) * cdf_gamma(k - 1,
                                                                                             shape,
                                                                                             scale)
        si_pmf = si_pmf + shape * scale * (2 * cdf_gamma(k - 1,
                                                         shape + 1,
                                                         scale) - cdf_gamma(k - 2,
                                                                            shape + 1,
                                                                            scale) - cdf_gamma(k,
                                                                                               shape + 1,
                                                                                               scale))
        si_pmf = np.array([np.max([0, i]) for i in si_pmf])

        self.si_pmf = si_pmf

    def compute_overall_infectivity(self):

        def fill_up_with_zeros(x, ref):
            x_nrows, ref_nrows = x.shape[0], ref.shape[0]
            updated_x = x
            if x_nrows < ref_nrows:
                updated_x = np.concatenate([x, np.zeros(1 + ref_nrows - x_nrows)])
            return updated_x

        incid, si_pmf = self.incidence, self.si_pmf
        si_pmf = fill_up_with_zeros(x=si_pmf, ref=incid)
        number_of_time_points = incid.shape[0]
        overall_infectivity = np.zeros((number_of_time_points,))
        for t in range(1, number_of_time_points + 1):
            overall_infectivity[t - 1] = (si_pmf[:t] * incid.iloc[:t][::-1]['incidence']).sum()
        overall_infectivity[0] = np.nan

        return overall_infectivity

    def compute_posterior_parameters(self, prior_shape=None, prior_scale=None):
        incid, si_pmf = self.incidence, self.si_pmf
        t_start, t_end = self.t_start, self.t_end
        if prior_shape is None:
            prior_shape = self.prior_shape
        if prior_scale is None:
            prior_scale = self.prior_scale

        number_of_time_windows = len(t_start)
        overall_infectivity = self.compute_overall_infectivity()
        final_mean_si = (si_pmf * range(len(si_pmf))).sum()

        posterior_shape = np.zeros(number_of_time_windows)
        posterior_scale = np.zeros(number_of_time_windows)

        for t in range(number_of_time_windows):
            if t_end[t] > final_mean_si:
                posterior_shape[t] = prior_shape + (incid.iloc[range(t_start[t], t_end[t] + 1)]["incidence"]).sum()
            else:
                posterior_shape[t] = np.nan

        for t in range(number_of_time_windows):
            if t_end[t] > final_mean_si:
                period_overall_infectivity = (overall_infectivity[range(t_start[t], t_end[t] + 1)]).sum()
                posterior_scale[t] = 1 / ((1 / prior_scale) + period_overall_infectivity)
            else:
                posterior_scale[t] = np.nan

        self.posterior_parameters['shape'] = posterior_shape
        self.posterior_parameters['scale'] = posterior_scale

    def sample_from_posterior(self, sample_size=1000):
        if not all([i in self.posterior_parameters.keys() for i in ['scale', 'shape']]):
            txt = "Can't sample from posterior before computing posterior parameters."
            raise IndexError(txt)
        posterior_shape = self.posterior_parameters['shape']
        posterior_scale = self.posterior_parameters['scale']
        number_of_time_windows = len(self.t_start)
        sample_r_posterior = np.zeros((number_of_time_windows, sample_size))
        for t in range(number_of_time_windows):
            if not t > len(posterior_shape) - 1:
                sample_r_posterior[t, ] = np.random.gamma(shape=posterior_shape[t],
                                                          scale=posterior_scale[t],
                                                          size=sample_size)
            else:
                sample_r_posterior[t,] = np.nan

        return sample_r_posterior.transpose()

    def compute_posterior_summaries(self, posterior_sample, t_max=None):
        start_dates = self.incidence.index[self.t_start]
        end_dates = self.incidence.index[self.t_end]
        post_mean_r = posterior_sample.mean(axis=0)
        post_sd = posterior_sample.std(axis=0)
        post_shape = self.posterior_parameters['shape']
        post_scale = self.posterior_parameters['scale']
        post_upper_quantile_r = np.quantile(posterior_sample, q=0.975, axis=0)
        post_lower_quantile_r = np.quantile(posterior_sample, q=0.025, axis=0)
        summary_dict = {
            'start_dates': start_dates, 'end_dates': end_dates,
            'Rt_mean': post_mean_r, 'Rt_sd': post_sd,
            'Rt_q0.975': post_upper_quantile_r, 'Rt_q0.025': post_lower_quantile_r,
            'Rt_shape': post_shape, 'Rt_scale': post_scale
        }
        posterior_summary = pd.DataFrame(summary_dict)
        posterior_summary['start_dates'] = posterior_summary['start_dates'].astype('datetime64[ns]')
        posterior_summary['end_dates'] = posterior_summary['end_dates'].astype('datetime64[ns]')

        if t_max is not None:
            last_day = max(posterior_summary['end_dates'])
            final_date = max(posterior_summary['end_dates']) + pd.Timedelta(days=t_max)
            last_day_data = posterior_summary[posterior_summary['end_dates'] == last_day].to_dict(orient='list')
            dates_ahead = pd.date_range(start=last_day, end=final_date)[1:]

            forecast_d = pd.DataFrame({
                'start_dates': pd.NaT, 'end_dates': dates_ahead
            })

            forecast_d['Rt_mean'] = last_day_data['Rt_mean'][0]
            forecast_d['Rt_sd'] = last_day_data['Rt_sd'][0]
            forecast_d['Rt_q0.975'] = last_day_data['Rt_q0.975'][0]
            forecast_d['Rt_q0.025'] = last_day_data['Rt_q0.025'][0]
            forecast_d['Rt_shape'] = last_day_data['Rt_shape'][0]
            forecast_d['Rt_scale'] = last_day_data['Rt_scale'][0]

            posterior_summary = pd.concat([posterior_summary, forecast_d], ignore_index=True)
            posterior_summary['estimation_type'] = np.where(posterior_summary['end_dates'] <= last_day,
                                                            'fitted', 'forecasted')

        self.posterior_summary = posterior_summary

    def plot_reproduction_number(self, title=None, filename=None):
        d = self.posterior_summary
        if d is None:
            txt = "You need to compute the summaries for the posterior distribution of Rt."
            raise ValueError(txt)
        if title is None:
            title = "R(t): time-varying reproduction number"
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.plot(d['end_dates'], d['Rt_mean'], color='b')
        plt.plot(d['end_dates'], [1] * len(d['Rt_mean']), color='gray', linestyle='dashed', alpha=0.75)
        plt.fill_between(d['end_dates'],
                         d['Rt_q0.975'],
                         d['Rt_q0.025'],
                         color='b', alpha=0.2)
        plt.title(title)
        plt.suptitle("$P(R_t | Data) \sim Gamma(k_t, \\theta_t)$")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        fig.autofmt_xdate()

        if 'estimation_type' in d.columns:
            plt.axvline(x=max(d[d['estimation_type'] == "fitted"]["end_dates"]),
                        color='gray', linestyle='dashed', alpha=0.75)

        if filename is None:
            plt.show()
        else:
            fig.savefig(filename, dpi=fig.dpi)
            plt.close()