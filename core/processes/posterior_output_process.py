import os
import json
import numpy as np
import KratosMultiphysics as Kratos


def Factory(forward_model, likelihood, settings):
    return PosteriorOutputProcess(forward_model, settings)


class PosteriorOutputProcess:
    """Saves the posterior samples themselves: .npz always, summary.json, and an
    Excel workbook if pandas/openpyxl are available."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module" : "posterior_output_process",
            "Parameters"    : {
                "output_path"  : "output",
                "write_npz"    : true,
                "write_excel"  : true
            }
        }""")

    def __init__(self, forward_model, settings):
        settings.ValidateAndAssignDefaults(self.GetDefaultParameters()["Parameters"])
        self.model = forward_model
        self.path = settings["output_path"].GetString()
        self.write_npz = settings["write_npz"].GetBool()
        self.write_excel = settings["write_excel"].GetBool()

    def ExecuteFinalize(self, levels, q, logcE):
        os.makedirs(self.path, exist_ok=True)
        posterior = levels[-1]
        refs = self.model.refs

        if self.write_npz:
            np.savez(os.path.join(self.path, "smc_levels.npz"),
                     q=q, logcE=logcE, E_ref=refs,
                     prop_ids=np.array([ids[0] for ids in self.model.prop_ids]),
                     E_posterior=posterior * refs,
                     **{f"level_{i:02d}": lv for i, lv in enumerate(levels)})

        stats = [{"zone": z + 1,
                  "alpha_mean": float(posterior[:, z].mean()),
                  "alpha_std": float(posterior[:, z].std(ddof=1)),
                  "alpha_p2.5": float(np.percentile(posterior[:, z], 2.5)),
                  "alpha_p97.5": float(np.percentile(posterior[:, z], 97.5)),
                  "E_ref_Pa": float(refs[z]),
                  "E_mean_Pa": float(posterior[:, z].mean() * refs[z]),
                  "E_std_Pa": float(posterior[:, z].std(ddof=1) * refs[z])}
                 for z in range(posterior.shape[1])]

        with open(os.path.join(self.path, "summary.json"), "w") as f:
            json.dump({"logcE": logcE, "tempering_q": q.tolist(),
                       "n_particles": int(len(posterior)),
                       "n_forward_solves": self.model.n_solves,
                       "zones": stats}, f, indent=2)

        if self.write_excel:
            self._Excel(levels, q, refs, stats)

        for s in stats:
            Kratos.Logger.PrintInfo(
                "PosteriorOutput",
                "zone %d: alpha = %.4f +/- %.4f   E = %.4e +/- %.3e Pa"
                % (s["zone"], s["alpha_mean"], s["alpha_std"],
                   s["E_mean_Pa"], s["E_std_Pa"]))

    def _Excel(self, levels, q, refs, stats):
        try:
            import pandas as pd
            import openpyxl        # pandas imports it lazily, so check it here
        except ImportError:
            Kratos.Logger.PrintWarning("PosteriorOutput",
                                       "pandas/openpyxl missing - skipping xlsx")
            return
        with pd.ExcelWriter(os.path.join(self.path, "posterior.xlsx"),
                            engine="openpyxl") as writer:
            for lev, samples in enumerate(levels):
                cols = {}
                for z in range(samples.shape[1]):
                    cols[f"alpha_{z + 1}"] = samples[:, z]
                    cols[f"E_{z + 1}_Pa"] = samples[:, z] * refs[z]
                pd.DataFrame(cols).to_excel(
                    writer, sheet_name=f"level_{lev:02d}_q{q[lev]:.3f}"[:31], index=False)
            pd.DataFrame(stats).to_excel(writer, sheet_name="posterior_stats", index=False)
