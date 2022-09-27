# QMC-for-bond-and-current-interactions
Projector determinant quantum Monte Carlo algorithm for dealing with the singlet bond and triplet current interactions

The SU(*N*) model Hamiltonian with singlet-bond and triplet-current interactions is defined as [1]

$$ H = -t\sum_{\langle{ij}\rangle,\alpha}(c_{i,\alpha}^{\dagger}c_{j,\alpha} +h.c.) + \sum_{\langle{ij}\rangle}[-\frac{g_{1}}{2}M_{ij}^2-\frac{g_{2}}{2}\vec{N}_{ij}^2]. $$

where the generalized SU(*N*)-symmetric singlet bond operator,

$$ M_{ij} = \frac{1}{\sqrt{N}}\sum_{\alpha=1}^{N}(c_{i,\alpha}^{\dagger}\sigma_0c_{j,\alpha}+h.c.), $$

and triplet current operator,

$$ \vec{N}_{ij} = \frac{i}{\sqrt{N}}\sum_{\alpha=1}^{N}(c_{i,\alpha}^{\dagger}\frac{\vec{\sigma}}{2}c_{j,\alpha}-h.c.). $$

In PQMC, the Kramer's time-reversal invariant decomposition is used [1],

$$ e^{\Delta\tau gM_{ij}^2} = \sum_{l,s=\pm1}\frac{\gamma_l}{4}e^{s\eta_l\sqrt{\Delta\tau g}M_{ij}}+\mathcal{O}(\Delta\tau^4),\quad
   e^{\Delta\tau g\vec{N}_{ij}^2} = \prod_{a=x,y,z}\sum_{l_a,s_a=\pm1}\frac{\gamma_{l_a}}{4}e^{s_a\eta_{l_a}\sqrt{\Delta\tau g}{N}_{ij,a}}+\mathcal{O}(\Delta\tau^4), $$

where $\gamma_l = 1+\frac{\sqrt{6}}{3}l$ and $\eta_l=\sqrt{2(3-\sqrt{6}l)}$.

The lattice bonds of a square lattice are categorized into four groups, so that the operators, $M_{ij}$, within each group are commute.



[1]  C. Wu and S.-C. Zhang, Phys. Rev. B 71, 155115 (2005)

----

