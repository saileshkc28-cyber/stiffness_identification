from samplers.smc_acs_sampler import SMCaCSSampler


def Factory(likelihood, prior, settings):
    return SMCaCSMultiZoneSampler(likelihood, prior, settings)


class SMCaCSMultiZoneSampler(SMCaCSSampler):
    """SMC_aCS returns each level as [dim, n_particles]. The base class reshapes it
    to [n_particles, dim] without transposing, which mixes the zones when dim > 1.
    This undoes that reshape and transposes instead."""

    def Run(self):
        super().Run()
        dim = self.prior.dim
        self.levels = [x.ravel().reshape(dim, -1).T for x in self.levels]
        return self.levels
