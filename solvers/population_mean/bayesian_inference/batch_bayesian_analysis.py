"""Batch controller: one independent Bayesian inversion per Phase 1 realization.

Each Phase 1 sample carries its OWN latent Young's modulus E_i = mu_E + sigma_E * xi_i.
The measurements in sample_XXXX/measured_data.csv were generated from that single E_i,
so each inversion is a separate, self-contained inference problem:

    p(alpha_i | u_hat_i),    E_i = alpha_i * E_ref

This is a batch of independent inversions, NOT a hierarchical model. Nothing is shared
between samples except the FE model definition and the prior. In particular:

  * the prior stays the broad uniform prior from BayesianParameters.json. It is NOT
    replaced by the Phase 1 normal, because that normal is exactly the object the batch
    is trying to recover -- handing it to the sampler would beg the question;
  * E_true / alpha_true are read ONLY after an inversion finishes, for validation
    plots and error columns. They never reach the likelihood or the sampler.

Two different spreads are reported and must not be confused:

  within-posterior sd   uncertainty about alpha_i given one noisy dataset
  between-sample sd     variability of E between Phase 1 realizations  <- the target

The recovered population sd is the sd of the per-case posterior MEANS, not the sd of
all posterior draws pooled together (which would add the two spreads above).
"""
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import KratosMultiphysics as Kratos

from bayesian_analysis import BayesianAnalysis


def GetDefaultParameters():
    return Kratos.Parameters("""{
        "enabled"            : false,
        "phase1_root"        : "../damaged_system/phase1_distribution_runs",
        "aggregate_file"     : "phase1_samples.csv",
        "output_path"        : "output_distribution",
        "start_sample"       : 1,
        "end_sample"         : 0,
        "max_samples"        : 0,
        "resume"             : true,
        "base_random_seed"   : 20260802,
        "n_resample_repeats" : 1000,
        "make_plots"         : true
    }""")


# --------------------------------------------------------------------------- helpers
def one_line(exc):
    """Concise single-line exception message for the CSV and the failure log."""
    text = f"{type(exc).__name__}: {exc}"
    return " ".join(text.split())[:400]


def read_phase1_index(root, aggregate_file):
    """sample_id -> {status, xi, E_true, alpha_true} from Phase 1's aggregate CSV."""
    path = os.path.join(root, aggregate_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Phase 1 aggregate file not found: {path}")

    index = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = int(row["sample_id"])
            index[sid] = {
                "status": row.get("status", "").strip().strip('"'),
                "xi": float(row["xi"]),
                "E_true": float(row["E_true"]),
                "alpha_true": float(row["alpha_true"]),
            }
    return index


def read_measured_values(path):
    """The same column the existing Likelihood reads, so the two never disagree."""
    with open(path, newline="") as f:
        values = np.array([float(row["value"]) for row in csv.DictReader(f)])
    if values.size == 0:
        raise ValueError(f"no measurement rows in {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite measurement value in {path}")
    return values


def read_sample_noise(sample_dir):
    """sigma for this realization: noise_model.json first, metadata as fallback."""
    for name, key in (("noise_model.json", "sigma"), ("measurement_metadata.json", "sigma")):
        path = os.path.join(sample_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                value = json.load(f).get(key)
            if value is not None:
                sigma = float(value)
                if not np.isfinite(sigma) or sigma <= 0.0:
                    raise ValueError(f"sigma = {sigma} in {path}; a positive value is required")
                return sigma
    raise FileNotFoundError(f"no noise_model.json or measurement_metadata.json in {sample_dir}")


def read_sample_metadata(sample_dir):
    path = os.path.join(sample_dir, "measurement_metadata.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def rebase_output_path(original, sample_output_dir):
    """'output/vtk_output' -> '<sample_dir>/vtk_output';  'output' -> '<sample_dir>'.

    Keeps whatever sub-structure the existing output processes expect while making the
    destination unique, so no sample can overwrite another sample's results.
    """
    parts = [p for p in original.replace("\\", "/").split("/") if p not in ("", ".")]
    return os.path.join(sample_output_dir, *parts[1:]) if len(parts) > 1 else sample_output_dir


def build_sample_parameters(base_parameters, measured_file, sigma, seed, sample_output_dir):
    """Clone the single-dataset parameters and retarget them at one realization."""
    parameters = base_parameters.Clone()

    # BayesianAnalysis validates against its own defaults, which know nothing about
    # the batch block, so it has to go before the clone is handed over.
    if parameters.Has("batch_inference"):
        parameters.RemoveValue("batch_inference")

    parameters["likelihood"]["measured_data_file"].SetString(measured_file)
    parameters["likelihood"]["noise_model"]["sigma"].SetDouble(float(sigma))
    parameters["sampler_settings"]["Parameters"]["random_seed"].SetInt(int(seed))

    for i in range(parameters["output_processes"].size()):
        settings = parameters["output_processes"][i]["Parameters"]
        if settings.Has("output_path"):
            settings["output_path"].SetString(
                rebase_output_path(settings["output_path"].GetString(), sample_output_dir))

    return parameters


def summarize(alpha_samples, E_ref):
    E_samples = E_ref * alpha_samples
    summary = {}
    for tag, data in (("alpha", alpha_samples), ("E", E_samples)):
        summary[f"{tag}_posterior_mean"] = float(np.mean(data))
        summary[f"{tag}_posterior_sd"] = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
        summary[f"{tag}_posterior_median"] = float(np.median(data))
        summary[f"{tag}_ci_2_5"] = float(np.percentile(data, 2.5))
        summary[f"{tag}_ci_97_5"] = float(np.percentile(data, 97.5))
    return summary, E_samples


# ----------------------------------------------------------------------- one inversion
def run_single_inference(sample_id, sample_dir, sample_output_dir, base_parameters, seed):
    """One complete, independent inversion. Returns the summary dict for this sample."""
    measured_file = os.path.join(sample_dir, "measured_data.csv")
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"sample directory missing: {sample_dir}")
    if not os.path.isfile(measured_file):
        raise FileNotFoundError(f"measured_data.csv missing: {measured_file}")

    u_hat = read_measured_values(measured_file)          # finiteness checked inside
    sigma = read_sample_noise(sample_dir)                 # sigma > 0 enforced inside

    os.makedirs(sample_output_dir, exist_ok=True)
    parameters = build_sample_parameters(base_parameters, measured_file, sigma,
                                         seed, sample_output_dir)

    # a fresh Model per realization: sharing one would leak model part state and
    # trigger the "root modelpart already exists" path in Kratos
    model = Kratos.Model()
    analysis = BayesianAnalysis(model, parameters)

    # Run() is split here only to insert the dimension and finiteness checks between
    # setup and the expensive sampling; the sequence is otherwise identical
    analysis.Initialize()

    n_sensors = len(analysis.forward_model.located)
    if u_hat.size != n_sensors:
        raise ValueError(f"measured vector has {u_hat.size} entries but the model has "
                         f"{n_sensors} sensors")
    if analysis.likelihood.u_hat.size != n_sensors:
        raise ValueError(f"likelihood loaded {analysis.likelihood.u_hat.size} values for "
                         f"{n_sensors} sensors")

    probe = np.asarray(analysis.forward_model.Evaluate([1.0]), dtype=float)
    if probe.size != n_sensors or not np.all(np.isfinite(probe)):
        raise ValueError("forward model returned a non-finite or wrongly sized response")
    analysis.forward_model.n_solves = 0        # do not bill the probe to this sample

    analysis.RunSolutionLoop()
    analysis.Finalize()

    E_ref = float(np.atleast_1d(analysis.forward_model.refs)[0])

    posterior = np.asarray(analysis.sampler.levels[-1], dtype=float)
    if posterior.ndim == 1:
        posterior = posterior.reshape(-1, 1)
    if posterior.size == 0:
        raise ValueError("empty final posterior level")
    alpha_samples = posterior[:, 0]
    if not np.all(np.isfinite(alpha_samples)):
        raise ValueError("non-finite posterior draws")

    summary, E_samples = summarize(alpha_samples, E_ref)
    summary.update({
        "sample_id": sample_id,
        "status": "ok",
        "number_of_posterior_samples": int(alpha_samples.size),
        "number_of_forward_solves": int(analysis.forward_model.n_solves),
        "number_of_levels": int(len(analysis.sampler.q)),
        "sigma_used": sigma,
        "random_seed": int(seed),
        "measured_data_file": measured_file,
        "E_ref": E_ref,
    })

    np.savez_compressed(os.path.join(sample_output_dir, "posterior_samples.npz"),
                        alpha=alpha_samples, E=E_samples,
                        q=np.asarray(analysis.sampler.q, dtype=float),
                        E_ref=np.array([E_ref]))
    with open(os.path.join(sample_output_dir, "posterior_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del analysis, model
    gc.collect()
    return summary


# ----------------------------------------------------------- population postprocessing
def resample_population(alpha_draw_sets, E_ref, n_repeats, seed):
    """Propagate within-posterior uncertainty into the population statistics.

    One draw is taken from each case's posterior, then the mean and sample sd of that
    synthetic population are recorded. Repeating this gives an uncertainty band for the
    recovered mean and sd WITHOUT pooling all draws (which would inflate the sd by
    adding within-posterior spread to between-sample spread).
    """
    if len(alpha_draw_sets) < 2:
        return None

    rng = np.random.default_rng(seed)
    means, sds = np.empty(n_repeats), np.empty(n_repeats)
    sizes = np.array([len(a) for a in alpha_draw_sets])
    for r in range(n_repeats):
        picks = np.array([a[i] for a, i in
                          zip(alpha_draw_sets, rng.integers(0, sizes))])
        means[r] = picks.mean()
        sds[r] = picks.std(ddof=1)

    return {
        "alpha_population_mean_ci_95": [float(np.percentile(means, 2.5)),
                                        float(np.percentile(means, 97.5))],
        "alpha_population_sd_ci_95": [float(np.percentile(sds, 2.5)),
                                      float(np.percentile(sds, 97.5))],
        "E_population_mean_ci_95": [float(E_ref * np.percentile(means, 2.5)),
                                    float(E_ref * np.percentile(means, 97.5))],
        "E_population_sd_ci_95": [float(E_ref * np.percentile(sds, 2.5)),
                                  float(E_ref * np.percentile(sds, 97.5))],
        "n_resample_repeats": int(n_repeats),
    }


def make_plots(records, target_mean, target_sd, output_root):
    """Validation figures. Every one of these uses truth values, which is legitimate
    here because inference has already finished."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plots")
        return []

    ok = [r for r in records if r["status"] == "ok" and np.isfinite(r.get("E_true", np.nan))]
    if len(ok) < 2:
        print("not enough successful samples to plot")
        return []

    E_true = np.array([r["E_true"] for r in ok])
    E_mean = np.array([r["E_posterior_mean"] for r in ok])
    E_lo = np.array([r["E_ci_2_5"] for r in ok])
    E_hi = np.array([r["E_ci_97_5"] for r in ok])
    error = E_mean - E_true
    written = []

    def save(fig, name):
        path = os.path.join(output_root, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 1 -- do the recovered values reproduce the Phase 1 population?
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = min(30, max(6, len(ok) // 3))
    ax.hist(E_true / 1e9, bins=bins, density=True, alpha=0.55, label="Phase 1 $E_{true}$")
    ax.hist(E_mean / 1e9, bins=bins, density=True, alpha=0.55,
            label="recovered posterior means")
    ax.set_xlabel("E [GPa]")
    ax.set_ylabel("density")
    ax.set_title("between-sample variability: truth vs recovery")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "recovered_vs_true_hist.png")

    # 2 -- per-case accuracy
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [min(E_true.min(), E_mean.min()) / 1e9, max(E_true.max(), E_mean.max()) / 1e9]
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1")
    ax.scatter(E_true / 1e9, E_mean / 1e9, s=18, alpha=0.7)
    ax.set_xlabel("$E_{true}$ [GPa]")
    ax.set_ylabel("$E$ posterior mean [GPa]")
    ax.set_title("per-realization recovery")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "recovery_scatter.png")

    # 3 -- the same, with the WITHIN-posterior uncertainty drawn as bars
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1")
    ax.errorbar(E_true / 1e9, E_mean / 1e9,
                yerr=[(E_mean - E_lo) / 1e9, (E_hi - E_mean) / 1e9],
                fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.7,
                label="95% posterior CI (within-sample)")
    ax.set_xlabel("$E_{true}$ [GPa]")
    ax.set_ylabel("$E$ posterior mean [GPa]")
    ax.set_title("bars = uncertainty within one posterior;\n"
                 "spread along the diagonal = variability between realizations",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "recovery_scatter_ci.png")

    # 4 -- bias and scatter of the recovery error
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(error / 1e9, bins=bins, density=True, edgecolor="black", alpha=0.75)
    ax.axvline(0.0, color="k", ls="--", lw=1.2)
    ax.axvline(error.mean() / 1e9, color="crimson", lw=1.5,
               label=f"mean error = {error.mean() / 1e9:.3f} GPa")
    ax.set_xlabel(r"$E_{posterior\ mean} - E_{true}$ [GPa]")
    ax.set_ylabel("density")
    ax.set_title("recovery error")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "recovery_error_hist.png")

    # 5 -- target normal vs the normal implied by the recovered population statistics
    rec_mean, rec_sd = E_mean.mean(), E_mean.std(ddof=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(min(E_true.min(), E_mean.min()), max(E_true.max(), E_mean.max()), 500)

    def pdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))

    if target_sd and target_sd > 0:
        ax.plot(grid / 1e9, pdf(grid, target_mean, target_sd) * 1e9, "k-", lw=2,
                label=f"Phase 1 target N({target_mean / 1e9:.1f}, {target_sd / 1e9:.1f}) GPa")
    ax.plot(grid / 1e9, pdf(grid, rec_mean, rec_sd) * 1e9, "r--", lw=2,
            label=f"recovered N({rec_mean / 1e9:.1f}, {rec_sd / 1e9:.1f}) GPa")
    ax.hist(E_mean / 1e9, bins=bins, density=True, alpha=0.35, label="posterior means")
    ax.set_xlabel("E [GPa]")
    ax.set_ylabel("density")
    ax.set_title("recovered population distribution vs Phase 1 target")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "target_vs_recovered_normal.png")

    return written


# ------------------------------------------------------------------------ batch driver
def apply_cli_overrides(settings):
    """--start N  --end N  --max-samples N  --no-plots  --no-resume  (smoke testing)."""
    argv = sys.argv[1:]
    for flag, key in (("--start", "start_sample"), ("--end", "end_sample"),
                      ("--max-samples", "max_samples")):
        if flag in argv:
            settings[key].SetInt(int(argv[argv.index(flag) + 1]))
    if "--no-plots" in argv:
        settings["make_plots"].SetBool(False)
    if "--no-resume" in argv:
        settings["resume"].SetBool(False)


def RunBatch(project_parameters):
    settings = project_parameters["batch_inference"]
    settings.ValidateAndAssignDefaults(GetDefaultParameters())
    apply_cli_overrides(settings)

    root = settings["phase1_root"].GetString()
    output_root = settings["output_path"].GetString()
    base_seed = settings["base_random_seed"].GetInt()
    resume = settings["resume"].GetBool()
    os.makedirs(output_root, exist_ok=True)

    index = read_phase1_index(root, settings["aggregate_file"].GetString())

    start = settings["start_sample"].GetInt()
    end = settings["end_sample"].GetInt()
    end = max(index) if end <= 0 else end
    max_samples = settings["max_samples"].GetInt()

    # only realizations Phase 1 itself marked valid
    selected = [sid for sid in sorted(index)
                if start <= sid <= end and index[sid]["status"] == "ok"]
    skipped_status = [sid for sid in sorted(index)
                      if start <= sid <= end and index[sid]["status"] != "ok"]
    if max_samples > 0:
        selected = selected[:max_samples]

    print(f"\nPhase 1 root      : {root}")
    print(f"selected samples  : {len(selected)}  (range {start}-{end}, "
          f"{len(skipped_status)} rejected by Phase 1 status)")
    print(f"output root       : {output_root}\n")

    failure_log = os.path.join(output_root, "phase2_failures.log")
    records = []
    t_start = time.time()

    for n, sample_id in enumerate(selected, 1):
        sample_dir = os.path.join(root, f"sample_{sample_id:04d}")
        sample_output_dir = os.path.join(output_root, f"sample_{sample_id:04d}")
        summary_file = os.path.join(sample_output_dir, "posterior_summary.json")
        truth = index[sample_id]

        if resume and os.path.isfile(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
            summary.setdefault("status", "ok")
            print(f"[{n}/{len(selected)}] sample {sample_id:04d}: resumed")
        else:
            seed = base_seed + sample_id          # reproducible, distinct per realization
            print(f"[{n}/{len(selected)}] sample {sample_id:04d}: seed {seed}", flush=True)
            t0 = time.time()
            try:
                summary = run_single_inference(sample_id, sample_dir, sample_output_dir,
                                               project_parameters, seed)
                print(f"    alpha = {summary['alpha_posterior_mean']:.4f} "
                      f"+- {summary['alpha_posterior_sd']:.4f}   "
                      f"({summary['number_of_forward_solves']} solves, "
                      f"{time.time() - t0:.0f} s)")
            except Exception as exc:              # one bad sample must not stop the batch
                message = one_line(exc)
                summary = {"sample_id": sample_id, "status": "failed",
                           "error_message": message}
                with open(failure_log, "a") as f:
                    f.write(f"sample_{sample_id:04d}\t{message}\n")
                print(f"    FAILED: {message}")

        summary.update({"alpha_true": truth["alpha_true"], "E_true": truth["E_true"],
                        "xi": truth["xi"]})
        records.append(summary)

    # ------------------------------------------------------------------ aggregation
    ok = [r for r in records if r["status"] == "ok"]

    with open(os.path.join(output_root, "phase2_recovered_samples.csv"), "w",
              newline="") as f:
        columns = ["sample_id", "status", "alpha_posterior_mean", "alpha_posterior_sd",
                   "alpha_ci_2_5", "alpha_ci_97_5", "E_posterior_mean", "E_posterior_sd",
                   "E_ci_2_5", "E_ci_97_5", "alpha_true", "E_true", "alpha_error",
                   "E_error", "covered_by_95_percent_interval",
                   "number_of_forward_solves", "error_message"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            if r["status"] == "ok":
                # truth-based columns: diagnostics only, computed after inference
                row["alpha_error"] = r["alpha_posterior_mean"] - r["alpha_true"]
                row["E_error"] = r["E_posterior_mean"] - r["E_true"]
                row["covered_by_95_percent_interval"] = int(
                    r["alpha_ci_2_5"] <= r["alpha_true"] <= r["alpha_ci_97_5"])
            writer.writerow(row)

    metadata = read_sample_metadata(os.path.join(root, f"sample_{selected[0]:04d}")) \
        if selected else {}
    target_mean = float(metadata.get("E_distribution_mean", float("nan")))
    target_sd = float(metadata.get("E_distribution_sd", float("nan")))
    E_ref = float(ok[0]["E_ref"]) if ok else float(metadata.get("E_ref", float("nan")))

    population = {
        "requested_sample_count": len(selected),
        "successful_inference_count": len(ok),
        "failed_inference_count": len(records) - len(ok),
        "phase1_target_mean_E": target_mean,
        "phase1_target_sd_E": target_sd,
        "E_ref": E_ref,
    }

    if len(ok) >= 2:
        E_true = np.array([r["E_true"] for r in ok])
        E_mean = np.array([r["E_posterior_mean"] for r in ok])
        a_true = np.array([r["alpha_true"] for r in ok])
        a_mean = np.array([r["alpha_posterior_mean"] for r in ok])

        # PRIMARY recovered population: the distribution of per-case posterior means.
        # Deliberately NOT the sd of all pooled draws.
        recovered_mean_E = float(np.mean(E_mean))
        recovered_sd_E = float(np.std(E_mean, ddof=1))

        population.update({
            "phase1_empirical_mean_E": float(E_true.mean()),
            "phase1_empirical_sd_E": float(E_true.std(ddof=1)),
            "phase1_empirical_mean_alpha": float(a_true.mean()),
            "phase1_empirical_sd_alpha": float(a_true.std(ddof=1)),
            "recovered_mean_E": recovered_mean_E,
            "recovered_sd_E": recovered_sd_E,
            "recovered_mean_alpha": float(np.mean(a_mean)),
            "recovered_sd_alpha": float(np.std(a_mean, ddof=1)),
            "relative_error_mean_E": float((recovered_mean_E - E_true.mean())
                                           / E_true.mean()),
            "relative_error_sd_E": float((recovered_sd_E - E_true.std(ddof=1))
                                         / E_true.std(ddof=1)),
            "mean_within_posterior_sd_E": float(np.mean([r["E_posterior_sd"] for r in ok])),
            "coverage_95": float(np.mean([
                r["alpha_ci_2_5"] <= r["alpha_true"] <= r["alpha_ci_97_5"] for r in ok])),
        })

        draw_sets = []
        for r in ok:
            npz = os.path.join(output_root, f"sample_{r['sample_id']:04d}",
                               "posterior_samples.npz")
            if os.path.isfile(npz):
                draw_sets.append(np.load(npz)["alpha"])
        band = resample_population(draw_sets, E_ref,
                                   settings["n_resample_repeats"].GetInt(), base_seed)
        if band:
            population.update(band)

        if settings["make_plots"].GetBool():
            for path in make_plots(records, target_mean, target_sd, output_root):
                print(f"plot: {path}")

    with open(os.path.join(output_root, "phase2_population_summary.json"), "w") as f:
        json.dump(population, f, indent=2)

    # ---------------------------------------------------------------------- report
    print("\n=============== Phase 2 batch summary ===============")
    print(f"successful / requested : {len(ok)} / {len(selected)}")
    print(f"wall clock             : {time.time() - t_start:.0f} s")
    if len(ok) >= 2:
        print(f"Phase 1 target         : mean {target_mean:.6e}  sd {target_sd:.6e}")
        print(f"Phase 1 empirical      : mean {population['phase1_empirical_mean_E']:.6e}  "
              f"sd {population['phase1_empirical_sd_E']:.6e}")
        print(f"recovered (post. means): mean {population['recovered_mean_E']:.6e}  "
              f"sd {population['recovered_sd_E']:.6e}")
        print(f"relative error         : mean {population['relative_error_mean_E']:+.4%}  "
              f"sd {population['relative_error_sd_E']:+.4%}")
        print(f"within-posterior sd    : {population['mean_within_posterior_sd_E']:.6e} "
              "(NOT the population sd)")
        print(f"95% CI coverage        : {population['coverage_95']:.3f} (nominal 0.950)")
    print(f"outputs                : {output_root}")
    return population
