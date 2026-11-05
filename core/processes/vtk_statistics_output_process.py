import numpy as np
import KratosMultiphysics as Kratos


class FieldStats:
    """Welford running mean/std. Never solve at the mean theta: u(E[a]) != E[u(a)]."""

    def __init__(self):
        self.n, self.mu, self.m2 = 0, None, None

    def Update(self, field):
        field = np.asarray(field, float)
        if self.mu is None:
            self.mu, self.m2 = np.zeros_like(field), np.zeros_like(field)
        self.n += 1
        delta = field - self.mu
        self.mu += delta / self.n
        self.m2 += delta * (field - self.mu)

    def Mean(self):
        return self.mu

    def Std(self):
        return np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.zeros_like(self.m2)


def Factory(forward_model, likelihood, settings):
    return VtkStatisticsOutputProcess(forward_model, likelihood, settings)


class VtkStatisticsOutputProcess:
    """Writes posterior field statistics per tempering level.

    VtkOutputProcess cannot be used: it only emits *registered* Kratos variables,
    and DISPLACEMENT_MEAN / YOUNG_MODULUS_STD are not registered. Registering them
    is a C++ change; until then this writes the same mesh with correct array names.
    """

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module" : "vtk_statistics_output_process",
            "Parameters"    : {
                "output_path"        : "output/vtk_output",
                "write_every_level"  : true
            }
        }""")

    def __init__(self, forward_model, likelihood, settings):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters()["Parameters"])
        self.model = forward_model
        self.likelihood = likelihood
        self.path = settings["output_path"].GetString()
        self.every_level = settings["write_every_level"].GetBool()

    def ExecuteFinalize(self, levels, q):
        import os
        os.makedirs(self.path, exist_ok=True)
        for lev, thetas in enumerate(levels):
            if not self.every_level and lev != len(levels) - 1:
                continue
            disp, E = self._Accumulate(thetas)
            self._Write(os.path.join(self.path, f"smc_level_{lev:02d}.vtk"), disp, E,
                        f"SMC_aCS level {lev}, q = {q[lev]:.4f}")
        disp, E = self._Accumulate(levels[-1])
        self._Write(os.path.join(self.path, "posterior.vtk"), disp, E,
                    "posterior statistics, q = 1")

    def _Accumulate(self, thetas):
        disp, E = FieldStats(), FieldStats()
        for theta in thetas:
            cached = self.likelihood.Cached(theta)
            if cached is not None:
                disp.Update(cached[1])
                E.Update(self.model.YoungsModulus(theta))
        return disp, E

    def _Write(self, path, disp_stats, E_stats, title):
        nodes, cells = self.model.nodes, self.model.cells
        with open(path, "w") as f:
            f.write(f"# vtk DataFile Version 4.2\n{title}\nASCII\n"
                    f"DATASET UNSTRUCTURED_GRID\nPOINTS {len(nodes)} double\n")
            f.writelines(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n" for p in nodes)
            f.write(f"\nCELLS {len(cells)} {4 * len(cells)}\n")
            f.writelines(f"3 {c[0]} {c[1]} {c[2]}\n" for c in cells)
            f.write(f"\nCELL_TYPES {len(cells)}\n" + "5\n" * len(cells))

            f.write(f"\nPOINT_DATA {len(nodes)}\n")
            for name, arr in (("DISPLACEMENT_MEAN", disp_stats.Mean()),
                              ("DISPLACEMENT_STD", disp_stats.Std())):
                f.write(f"VECTORS {name} double\n")
                f.writelines(f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n" for v in arr)

            f.write(f"\nCELL_DATA {len(cells)}\n")
            for name, arr in (("YOUNG_MODULUS_MEAN", E_stats.Mean()),
                              ("YOUNG_MODULUS_STD", E_stats.Std())):
                f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                f.writelines(f"{v:.10e}\n" for v in arr)
