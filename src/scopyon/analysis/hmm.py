import numpy

from hmmlearn.base import _BaseHMM
from hmmlearn.hmm import _check_and_set_gaussian_n_features
from hmmlearn import _utils


class FullPTHMM(_BaseHMM):
    r"""Hidden Markov Model for Particle Tracking.

    Args:
        n_components (int): Number of states.
        min_var (float, optional): Floor on the variance to prevent overfitting.
            Defaults to 1e-5.
        startprob_prior (array, optional):
            shape (n_components, ). Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.
        transmat_prior (array, optional):
            shape (n_components, n_components). Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.
        algorithm (string, optional):
            Decoder algorithm. Must be one of "viterbi" or`"map".
            Defaults to "viterbi".
        random_state (RandomState or an int seed, optional):
            A random number generator instance.
        n_iter (int, optional): Maximum number of iterations to perform.
        tol (float, optional):
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.
        verbose (bool, optional):
            When ``True`` per-iteration convergence reports are printed
            to :data:`sys.stderr`. You can diagnose convergence via the
            :attr:`monitor_` attribute.
        params (string, optional):
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'd' for diffusivities, 'm' for intensity means
            and 'v' for intensity variances. Defaults to all parameters.
        init_params (string, optional):
            Controls which parameters are initialized prior to
            training.  Can contain any combination of 's' for startprob,
            't' for transmat, 'd' for diffusivities, 'm' for intensity means
            and 'v' for intensity variances. Defaults to all parameters.

    Attributes:
        monitor\_  (ConvergenceMonitor):
            Monitor object used to check the convergence of EM.
        startprob\_  (array): shape (n_components, ).
            Initial state occupation distribution.
        transmat\_ (array): shape (n_components, n_components).
            Matrix of transition probabilities between states.
        diffusivities\_ (array): shape (n_components, 1).
            Diffusion constants for each state.
        intensity_means\_  (array): shape (n_components, 1).
            Mean parameters of intensity distribution for each state.
        intensity_vars\_ (array): shape (n_components, 1).
            Variance parameters of intensity distribution for each state.
    """

    def __init__(self, n_components=1,
                 min_var=1e-5,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stdmv", init_params="stdmv"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.min_var = min_var

    def _check(self):
        super()._check()

        self.diffusivities_ = numpy.asarray(self.diffusivities_)
        assert self.diffusivities_.shape == (self.n_components, 1)
        self.intensity_means_ = numpy.asarray(self.intensity_means_)
        assert self.intensity_means_.shape == (self.n_components, 1)
        self.intensity_vars_ = numpy.asarray(self.intensity_vars_)
        assert self.intensity_vars_.shape == (self.n_components, 1)
        self.n_features = 1

    def _generate_sample_from_state(self, state, random_state=None):
        D = self.diffusivities_[state]
        mean = self.intensity_means_[state]
        var = self.intensity_vars_[state]
        return numpy.hstack([
            numpy.sqrt(numpy.power(random_state.normal(scale=numpy.sqrt(2 * D), size=2), 2).sum(keepdims=True)),
            random_state.normal(loc=mean, scale=numpy.sqrt(var), size=(1, )),
            ])

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "d": nc * nf,
            "m": nc * nf,
            "v": nc * nf,
        }

    def _init(self, X, lengths=None):
        _check_and_set_gaussian_n_features(self, X)
        super()._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features

        if 'd' in self.init_params or not hasattr(self, "diffusivities_"):
            diffusivity_means = numpy.mean(X[:, [0]], axis=0) * 0.25
            variations = numpy.arange(1, self.n_components + 1)
            variations = variations / variations.sum()
            self.diffusivities_ = diffusivity_means * variations[:, numpy.newaxis]

        if 'm' in self.init_params or not hasattr(self, "intensity_means_"):
            from sklearn import cluster
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X[:, [1]])
            self.intensity_means_ = kmeans.cluster_centers_

        if 'v' in self.init_params or not hasattr(self, "intensity_vars_"):
            var = numpy.var(X[:, [1]].T) + self.min_var
            self.intensity_vars_ = numpy.tile([var], (self.n_components, 1))

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()

        stats['post'] = numpy.zeros(self.n_components)
        stats['obs1**2'] = numpy.zeros((self.n_components, 1))
        stats['obs2'] = numpy.zeros((self.n_components, 1))
        stats['obs2**2'] = numpy.zeros((self.n_components, 1))
        return stats

    def _compute_log_likelihood(self, X):
        D = self.diffusivities_
        mean = self.intensity_means_
        var = self.intensity_vars_
        # print("D=", D)
        # print("mean=", mean)
        # print("var=", var)
        if not all(var > 0):
            raise ValueError(f'Variance must be positive [{var}]')

        q1 = numpy.log(X[:, [0]] / (2 * D[:, 0])) - (X[:, [0]] ** 2 / (4 * D[:, 0]))
        q2 = -0.5 * numpy.log(2 * numpy.pi * var[:, 0]) - (X[:, [1]] - mean[:, 0]) ** 2 / (2 * var[:, 0])
        return q1 + q2

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if any(param in self.params for param in 'dmv'):
            stats['post'] += posteriors.sum(axis=0)
        if 'd' in self.params:
            stats['obs1**2'] += numpy.dot(posteriors.T, obs[:, [0]] ** 2)
        if 'm' in self.params:
            stats['obs2'] += numpy.dot(posteriors.T, obs[:, [1]])
        if 'v' in self.params:
            stats['obs2**2'] += numpy.dot(posteriors.T, obs[:, [1]] ** 2)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        denom = stats['post'][:, numpy.newaxis]
        if 'd' in self.params:
            self.diffusivities_ = 0.25 * stats['obs1**2'] / denom
        if 'm' in self.params:
            self.intensity_means_ = stats['obs2'] / denom
        if 'v' in self.params:
            self.intensity_vars_ = (
                stats['obs2**2'] - 2 * self.intensity_means_ * stats['obs2'] + self.intensity_means_ ** 2 * denom) / denom

class PTHMM(_BaseHMM):
    r"""Hidden Markov Model for Particle Tracking.

    Args:
        n_diffusivities (int): Number of diffusivity states.
        n_oligomers (int): Number of oligomeric states.
            n_components is equal to (n_diffusivities * n_oliogmers).
        min_var (float, optional): Floor on the variance to prevent overfitting.
            Defaults to 1e-5.
        startprob_prior (array, optional):
            shape (n_components, ). Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.
        transmat_prior (array, optional):
            shape (n_components, n_components). Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.
        algorithm (string, optional):
            Decoder algorithm. Must be one of "viterbi" or`"map".
            Defaults to "viterbi".
        random_state (RandomState or an int seed, optional):
            A random number generator instance.
        n_iter (int, optional): Maximum number of iterations to perform.
        tol (float, optional):
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.
        verbose (bool, optional):
            When ``True`` per-iteration convergence reports are printed
            to :data:`sys.stderr`. You can diagnose convergence via the
            :attr:`monitor_` attribute.
        params (string, optional):
            Controls which parameters are updated in the training
            process.  Can contain any combination of 's' for startprob,
            't' for transmat, 'd' for diffusivities, 'm' for intensity means
            and 'v' for intensity variances. Defaults to all parameters.
        init_params (string, optional):
            Controls which parameters are initialized prior to
            training.  Can contain any combination of 's' for startprob,
            't' for transmat, 'd' for diffusivities, 'm' for intensity means
            and 'v' for intensity variances. Defaults to all parameters.

    Attributes:
        monitor\_  (ConvergenceMonitor):
            Monitor object used to check the convergence of EM.
        startprob\_  (array): shape (n_components, ).
            Initial state occupation distribution.
        transmat\_ (array): shape (n_components, n_components).
            Matrix of transition probabilities between states.
        diffusivities\_ (array): shape (n_diffusivities, 1).
            Diffusion constants for each state.
        intensity_means\_  (array): shape (1, 1).
            Base mean parameter of intensity distributions.
        intensity_vars\_ (array): shape (1, 1).
            Base Variance parameter of intensity distributions.
    """

    def __init__(self, n_diffusivities=3, n_oligomers=4,
                 min_var=1e-5,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stdmv", init_params="stdmv"):
        _BaseHMM.__init__(self, n_diffusivities * n_oligomers,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.min_var = min_var
        self.n_diffusivities = n_diffusivities
        self.n_oligomers = n_oligomers
        assert self.n_components == self.n_diffusivities * self.n_oligomers

    def _check(self):
        super()._check()

        self.diffusivities_ = numpy.asarray(self.diffusivities_)
        assert self.diffusivities_.shape == (self.n_diffusivities, 1)
        self.intensity_means_ = numpy.asarray(self.intensity_means_)
        assert self.intensity_means_.shape == (1, 1)
        self.intensity_vars_ = numpy.asarray(self.intensity_vars_)
        assert self.intensity_vars_.shape == (1, 1)

        self.n_features = 2

    def _generate_sample_from_state(self, state, random_state=None):
        m = state // self.n_oligomers
        n = state % self.n_oligomers
        mean = self.intensity_means_[0] * (n + 1)
        var = self.intensity_vars_[0] * (n + 1)
        D = self.diffusivities_[m]
        return numpy.hstack([
            numpy.sqrt(numpy.power(random_state.normal(scale=numpy.sqrt(2 * D), size=2), 2).sum(keepdims=True)),
            random_state.normal(loc=mean, scale=numpy.sqrt(var), size=(1, )),
            ])

    def _get_n_fit_scalars_per_param(self):
        return {
            "s": self.n_components - 1,
            "t": self.n_components * (self.n_components - 1),
            "d": self.n_diffusivities,
            "m": 1,
            "v": 1,
        }

    def _init(self, X, lengths=None):
        _check_and_set_gaussian_n_features(self, X)
        super()._init(X, lengths=lengths)

        _, n_features = X.shape
        assert n_features == 2
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
        self.n_features = n_features

        if 'd' in self.init_params or not hasattr(self, "diffusivities_"):
            diffusivity_means = numpy.mean(X[:, [0]], axis=0) * 0.25
            variations = numpy.arange(1, self.n_diffusivities + 1)
            variations = variations / variations.sum()
            self.diffusivities_ = diffusivity_means * variations[:, numpy.newaxis]

        if 'm' in self.init_params or not hasattr(self, "intensity_means_"):
            # kmeans = cluster.KMeans(n_clusters=self.n_components,
            #                         random_state=self.random_state)
            # kmeans.fit(X[:, [1]])
            # self.intensity_means_ = kmeans.cluster_centers_
            self.intensity_means_ = numpy.array([[numpy.average(X[:, 1]) * 0.5]])

        if 'v' in self.init_params or not hasattr(self, "intensity_vars_"):
            var = numpy.var(X[:, [1]].T) + self.min_var
            self.intensity_vars_ = numpy.array([[var]])

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = numpy.zeros(self.n_components)
        stats['obs1**2'] = numpy.zeros((self.n_components, 1))
        stats['obs2'] = numpy.zeros((self.n_components, 1))
        stats['obs2**2'] = numpy.zeros((self.n_components, 1))
        return stats

    def _compute_log_likelihood(self, X):
        # D = self.diffusivities_
        D = numpy.repeat(self.diffusivities_, self.n_oligomers, axis=0)
        mean = self.intensity_means_[0, 0]
        mean *= numpy.tile(numpy.arange(1, self.n_oligomers + 1), (1, self.n_diffusivities)).T
        var = self.intensity_vars_[0, 0]
        var *= numpy.tile(numpy.arange(1, self.n_oligomers + 1), (1, self.n_diffusivities)).T
        if any(var <= 0.0):
            raise ValueError(f'Variance must be positive [{var}]')
        q1 = numpy.log(X[:, [0]] / (2 * D[:, 0])) - (X[:, [0]] ** 2 / (4 * D[:, 0]))
        q2 = -0.5 * numpy.log(2 * numpy.pi * var[:, 0]) - (X[:, [1]] - mean[:, 0]) ** 2 / (2 * var[:, 0])
        # print("mean=", mean)
        # print("var=", var)
        # print("self.intensity_means_.shape=", self.intensity_means_.shape)
        # print("self.intensity_vars_.shape=", self.intensity_vars_.shape)
        # print("q1.shape=", q1.shape)
        # print("q2.shape=", q2.shape)
        return q1 + q2

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if any(param in self.params for param in 'dmv'):
            stats['post'] += posteriors.sum(axis=0)
        if 'd' in self.params:
            stats['obs1**2'] += numpy.dot(posteriors.T, obs[:, [0]] ** 2)
        if 'm' in self.params:
            stats['obs2'] += numpy.dot(posteriors.T, obs[:, [1]])
        if 'v' in self.params:
            stats['obs2**2'] += numpy.dot(posteriors.T, obs[:, [1]] ** 2)

        # print("posteriors=", posteriors.shape)
        # print("obs=", obs.shape)
        # print("stats['post']=", stats['post'].shape)
        # print("stats['obs1**2']=", stats['obs1**2'].shape)
        # print("stats['obs2']=", stats['obs2'].shape)
        # print("stats['obs2**2']=", stats['obs2**2'].shape)
        # assert False

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        denom = stats['post'][:, numpy.newaxis]
        # print("denom=", denom.shape)
        # print("stats['post']=", stats['post'].shape)
        # print("stats['obs1**2']=", stats['obs1**2'].shape)
        # print("stats['obs2']=", stats['obs2'].shape)
        # print("stats['obs2**2']=", stats['obs2**2'].shape)
        # print("diffusivities_=", self.diffusivities_)
        # print("intensity_means_=", self.intensity_means_)
        # print("intensity_vars_=", self.intensity_vars_)

        if 'd' in self.params:
            k = numpy.repeat(numpy.identity(self.n_diffusivities), self.n_oligomers, axis=1)
            self.diffusivities_ = 0.25 * numpy.dot(k, stats['obs1**2']) / numpy.dot(k, denom)

        if 'm' in self.params:
            post = denom
            x = stats['obs2']
            k = numpy.tile(numpy.arange(1, self.n_oligomers + 1), (1, self.n_diffusivities))
            self.intensity_means_ = x.sum(axis=0) / numpy.dot(k, post)

        if 'v' in self.params:
            post = denom
            x = stats['obs2']
            x2 = stats['obs2**2']
            mu = self.intensity_means_
            k = numpy.tile(numpy.arange(1, self.n_oligomers + 1), (1, self.n_diffusivities))
            self.intensity_vars_ = (numpy.dot(1 / k, x2) - 2 * mu * x.sum(axis=0) + mu ** 2 * numpy.dot(k, post)) / post.sum(axis=0)
