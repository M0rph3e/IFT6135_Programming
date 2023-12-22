import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)
    # log_likelihood_bernoulli
    out = torch.sum(target * torch.log(mu) + (1-target)*torch.log(1-mu),1)
    return out


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # log normal
    sigma = torch.exp(logvar) #actually this is sigma^2

    #after some maths aside :
    r =  0.5 *  (z-mu)**2/sigma
    l = -torch.log(torch.sqrt(2*math.pi*sigma))
    return torch.sum(l-r,1)


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # log_mean_exp
    a = torch.max(y,1).values
    return torch.log(torch.sum(torch.exp(y.T - a).T,1)/sample_size) + a


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld
    #ref : https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    sigma_p = torch.exp(logvar_p) #those are sigma^2 !!
    sigma_q = torch.exp(logvar_q) #those are sigma^2 !!
    l = torch.log(torch.sqrt(sigma_p)/torch.sqrt(sigma_q)) #left term
    r = (sigma_q+(mu_q-mu_p)**2)/(2*sigma_p) #right term
    return torch.sum(l+r-0.5,1)


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # kld
    #ref : https://stats.stackexchange.com/questions/280885/estimate-the-kullback-leibler-kl-divergence-with-monte-carlo
    #samples num_samples values
    sigma_p = torch.exp(logvar_p) #those are sigma^2 !!
    sigma_q = torch.exp(logvar_q) #those are sigma^2 !!

    cov_q = torch.diag_embed(sigma_q)
    #print(cov_q.shape)
    #distrib =torch.distributions.multivariate_normal.MultivariateNormal(loc=mu_q,covariance_matrix=cov_q) #torch.distributions.Normal(mu_q,sigma_q))
    d = torch.distributions.normal.Normal(mu_q,sigma_q)
    #print("hmdl")
    x = d.rsample()
    #after some maths aside :
    r_q =  0.5 *  ((x-mu_q)**2/sigma_q)
    l_q = -torch.log(torch.sqrt(2*math.pi*sigma_q))
    log_q = l_q-r_q

    r_p =  0.5 *  ((x-mu_p)**2/sigma_p)
    l_p = -torch.log(torch.sqrt(2*math.pi*sigma_p))
    log_p = l_p-r_p
    return torch.mean(log_q-log_p)
