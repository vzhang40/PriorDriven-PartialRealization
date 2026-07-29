## Partial Realization for model reduction of linear Bayesian inverse problems

This Github repository contains code for the numerical experiments in the following work:

1. Zhang, V., König, J. [in progress] Partial Realization for model reduction of linear Bayesian inverse problems.

## Background

In the Bayseian inference approach (for linear models), model parameters $p \in ℝ^d$ are related to measurement data $m \in ℝ^{d_\text{obs}}$ through the following observation model:

$$
\begin{equation}
\mathbf{m} = G p + \epsilon.
\end{equation}
$$

where $G \in ℝ^{d_\text{obs} \times d}$ is the forward model and $\epsilon \in ℝ^{d_\text{obs}}$ is observation noise.

These numerical experiments consider the Bayesian inference setting where we assume the observation noise is Gaussian, i.e., $\epsilon \sim 𝒩(0, \Gamma_\text{obs})$, and we are given a Gaussian prior on the model parameter, i.e., $p \sim 𝒩(0, \Gamma_0)$. 

We are interested in the setting where measurements come from a linear dynamical system. Let $A \in ℝ^{d \times d}$ and $C \in ℝ^{d_\text{out} \times d}$ be the state and output matrix respectively. The goal is to infer the unknown initial condition as model parameter $p$:

$$
\begin{aligned}
&\dot{x}(t) = Ax(t), \qquad & x(0) = p, \\
&y(t) = Cx(t).
\end{aligned}
$$

Our $n$ measurement snapshots are defined at times $0 < t_1 < \dots < t_n$ such that:

$$
m_i = y(t_i) + \epsilon_i \quad \text{for} \quad i = 1, \dots, n
$$

where $\epsilon_i \sim 𝒩(0, \Gamma_\epsilon)$. The measurement vector $m$ is constructed as follows:

$$
m = \begin{bmatrix} m_1\\ 
\vdots \\
m_n \end{bmatrix} \in ℝ^{d_\text{obs}}
$$

such that $d_\text{obs} = n d_\text{out}$.

Using knowledge of the linear dynamical system, we can construct the forward operator and observation covariance:

$$
G = \begin{bmatrix} Ce^{A t_1}\\ 
\vdots \\
Ce^{A t_n}
\end{bmatrix} \in ℝ^{d_\text{obs} \times d}  \quad \text{and} \quad
\Gamma_\text{obs} = \begin{bmatrix} 
\Gamma_\epsilon   &          &                   \\
                  &   \ddots &                   \\
                  &          &   \Gamma_\epsilon 
\end{bmatrix} \in ℝ^{d_\text{obs} \times d_\text{obs}}.
$$

Using Bayes' rule, we obtain the posterior $p | m \sim 𝒩(\mu_\text{pos}, \Gamma_\text{pos})$:

$$
\mu_\text{pos} = \Gamma_\text{pos} G^\top \Gamma^{-1}_\text{obs} m \in ℝ^d \quad \text{and} \quad \Gamma_\text{pos} = \Gamma_0 - \Gamma_0 G^\top (\Gamma_\text{obs} + G \Gamma_0 G^\top)^{-1} G \Gamma_0 \in ℝ^{d \times d}.
$$

**Limitation:** The forward model $G$ is high-dimensional and can be expensive to evaluate; thus, we harness system-theoretic model reduction to allievate the computational burden of obtaining posterior quantities.

## Prior-Driven System
The work in [2] introduces the prior-driven system which allows for the system-theoretic model reduction to be performed on the linear dynamical system of interest. Let $\Gamma_0 = L_0 L_0^\top$ and we can reinterpret the initial condition as a impulse input $u(t) = \delta(t)$ to the following prior-driven system:

$$
\begin{aligned}
&\dot{x}(t) = Ax(t) + L_0 u(t), \qquad & x(0) = 0, \\
&y_\epsilon(t) = \Gamma^{-1/2}_\epsilon C x(t).
\end{aligned}
$$

### Model Reduction Methods
In this repository, we estimate posterior quantities using prior-driven balanced truncation [2] and prior-driven partial realization [1]. The work in [3] shows that the posterior estimation error can be attributed to how well the reduced order model estimates the impulse response of the prior-driven system. 

$$
\text{Impulse Response: } \quad h(t) = \Gamma^{-1/2}_\epsilon Ce^{A t} L_0
$$

These prior-driven model reduction methods can be briefly described as applying balanced truncation [4, page 211] and partial realization [4, page 346] to the prior-driven system [2] for posterior estimation.

## Examples

To run this code, you need the MATLAB Control System Toolbox. The code was tested on a machine equipped with 16 GB of unified memory and an Apple M5 chip. Computations were performed in MATLAB Version 26.1.0.3203278 (R2026a) running on macOS Tahoe 26.5, using Control System Toolbox Version 26.1 (R2026a).

### 1D Advection Diffusion Equation
The script `advec_diff_pdpr.m` generates plots and numerical results for the 1D advection-diffusion partial differential equation (PDE) example. Changing the parameters $a$ and $c$ in the following PDE allows for experimentation with different Hankel singular value decays.

The one-dimensional advection–diffusion equation is

$$
\frac{\partial x}{\partial t} = a \frac{\partial^2 x}{\partial \eta^2} - c \frac{\partial x}{\partial \eta}
$$

where

- $x(\eta,t)$ is the state variable,
- $a$ is the **diffusion coefficient**,
- $c$ is the **advection velocity**,
- $\eta$ is the spatial coordinate,
- $t$ is time.

### Additional Examples
The script `ex_pdpr.m` generators plots and numerical results for [benchmark model reduction examples](https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction) [5]. The examples included in the script are:

- 2D Heat Equation (modified from SLICOT `heat-cont.mat` example)
- Clamped Beam Model
- Building Model
- ISS 1R Model

The `.mat` files from [SLICOT](https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction) [5] are in the `models` folder.

## References 

2. König, J., Qian, E., & Freitag, M. A. (2026). Dimension and model reduction approaches for linear Bayesian inverse problems with rank-deficient prior covariances (arXiv:2506.23892). arXiv. https://doi.org/10.48550/arXiv.2506.23892
   
3. König, J., & Lie, H. C. (2026). Posterior error bounds for prior-driven balancing in linear Gaussian inverse problems (arXiv:2601.03971). arXiv. https://doi.org/10.48550/arXiv.2601.03971

4. Antoulas, A. C. (2005). Approximation of Large-Scale Dynamical Systems. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898718713

5. Younès Chahlaoui and Paul Van Dooren: [A collection of benchmark examples for model reduction of linear time invariant dynamical systems](https://www.slicot.org/objects/software/reports/SLWN2002-2.ps.gz); SLICOT Working Note 2002-2: February 2002.

### Acknowledgements

This project was funded as a [DAAD RISE Germany](https://www.daad.de/rise/en/rise-germany/) program internship, supervised by Josie König at the University of Potsdam ٩(^ᗜ^ )و.
