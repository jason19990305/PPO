import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

# define the function f(x)
def f(x):
    return 1/(1 + torch.exp(-x))

# Distribution parameters
mu_p = 3.5
mu_q = 1
sigma_p = 0.9
sigma_q = 1

# Create Normal distributions
p_dist = Normal(loc=mu_p, scale=sigma_p)  # p(x)
q_dist = Normal(loc=mu_q, scale=sigma_q)  # p(x)

# Plot PDFs of p(x) and q(x)
x = torch.linspace(-3, 6, 10000)
p_pdf = p_dist.log_prob(x).exp()
q_pdf = q_dist.log_prob(x).exp()


# f(x) = sin(x)
f_x = f(x)

expected_value_list = []
number_of_samples = []

# ------ importance sampling ------
for i in range(1000):
    q_sample_x = q_dist.sample((i * 100,)) # sample x by q(x)
    q_sample_f_x = f(q_sample_x)  # get f(x) by q(x)
    q_sample_p_prob = p_dist.log_prob(q_sample_x).exp()  # Get probability of p(x) at sampled x
    q_sample_q_prob = q_dist.log_prob(q_sample_x).exp()  # Get probability of q(x) at sampled x
    sample_ratio = q_sample_p_prob / q_sample_q_prob
    IS_expected_value = (q_sample_f_x * sample_ratio).mean()  # Importance sampling estimate of expected value
    expected_value_list.append(IS_expected_value.numpy())
    number_of_samples.append(i * 10)  # Store the number of samples used

# expected value of f(x) under p(x)
expected_value_p = (f(p_dist.sample((100000,)))).mean().numpy()


# Plot probability density functions
plt.plot(x.numpy(), p_pdf.numpy(), label='p(x)')
plt.plot(x.numpy(), q_pdf.numpy(), label='q(x)')
plt.title('Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()



# Plot f(x)
plt.plot(x.numpy(), f_x.numpy(), label='f(x)')
plt.title('f(x)')
plt.xlabel('x')
plt.ylabel('output')
plt.grid(True)
plt.legend()
plt.show()

# Importance Sampling Estimate Plot
plt.plot(number_of_samples, expected_value_list, label='Importance Sampling Estimate')
plt.axhline(y=expected_value_p, color='red', linestyle='--', label='Target')
plt.title('Expected Value of f(x) sampled by q(x)')
plt.xlabel('Number of Samples') 
plt.ylabel('Expected Value')
plt.grid(True)
plt.legend()
plt.show()
