"""Adjoint slopes du/dt at a few existing training stiffnesses.

For --n-points stiffnesses taken from the training CSV already on file, this runs
one primal solve and, per sensor, one adjoint solve through the Kratos
SystemIdentificationApplication, and records

    u      sensor value, read as BuildSurrogate.py reads it
    dudt   du/dt with t = ln(E / e_scale), from the adjoint YOUNG_MODULUS sensitivities
    ratio  dudt / (-u), which is 1 for a linear elastic model (u proportional to 1/E)

No finite differences are taken in this script.

Run from bayesian_inference/, as BuildSurrogate.py: the primal parameters use paths
relative to it.

Usage:
    python adjoint_training_solves.py
    python adjoint_training_solves.py --config <path to SurrogateParameters.json>
    python adjoint_training_solves.py --n-points 5 --overwrite
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# BuildSurrogate.py imports Kratos only inside build_forward_model, so this is safe
from BuildSurrogate import ExactResponse, load_done

ADJOINT_MODEL_PART = "AdjointStructure"
SENSOR_MODEL_PART = "AdjointSensors"
OUTPUT_NAME = "gradient_training_responses.csv"
RATIO_TOL = 1e-6        # |ratio - 1| allowed per sensor
E_SHARED_TOL = 1e-12    # E seen by the adjoint elements vs E set on the primal


# --------------------------------------------------------------------------
# Primal model
# --------------------------------------------------------------------------
def build_primal(base_config_path):
    """BuildSurrogate.build_forward_model, with the adjoint nodal variables added to
    the primal root model part before the mdpa is read.

    build_forward_model creates its own Kratos.Model inside, so it cannot be handed a
    pre-created model part; the steps below follow it line by line otherwise."""
    import KratosMultiphysics as Kratos
    import KratosMultiphysics.StructuralMechanicsApplication as SMA
    from kratos_forward_model import KratosForwardModel

    with open(base_config_path) as f:
        settings = Kratos.Parameters(f.read())

    for block in ("forward_model", "parameters"):
        if not settings.Has(block):
            raise RuntimeError(f"'{block}' missing from {base_config_path}")

    entries = [settings["parameters"][i] for i in range(settings["parameters"].size())]
    if len(entries) != 1:
        raise RuntimeError(
            f"this script is one-dimensional in E, but 'parameters' has {len(entries)} "
            "entries")

    # KratosForwardModel runs the same ValidateAndAssignDefaults; running it here first
    # only lets us read the primal parameters file before the model is built
    fm_settings = settings["forward_model"]
    fm_settings.ValidateAndAssignDefaults(KratosForwardModel.GetDefaultParameters())
    with open(fm_settings["primal_parameters_file"].GetString()) as f:
        primal = Kratos.Parameters(f.read())
    solver = primal["solver_settings"]
    root_name = solver["model_part_name"].GetString()

    # Hook: create the primal root model part before KratosForwardModel builds the
    # analysis. MechanicalSolver reuses an existing model part, and ImportMDPAModeler's
    # Model.CreateModelPart returns the existing one (with a warning), so the mdpa is
    # read into this part. MechanicalSolver sets DOMAIN_SIZE only on a part it creates
    # itself, so it is set here.
    model = Kratos.Model()
    root = model.CreateModelPart(root_name)
    root.ProcessInfo.SetValue(Kratos.DOMAIN_SIZE, solver["domain_size"].GetInt())

    # What StructuralMechanicsAdjointStaticSolver.AddVariables adds on top of
    # MechanicalSolver.AddVariables. The adjoint model part will share this variables
    # list and these nodes, and Kratos refuses to add a variable once nodes exist.
    root.AddNodalSolutionStepVariable(SMA.ADJOINT_DISPLACEMENT)
    if solver["rotation_dofs"].GetBool():
        root.AddNodalSolutionStepVariable(SMA.ADJOINT_ROTATION)
    root.AddNodalSolutionStepVariable(Kratos.SHAPE_SENSITIVITY)
    root.AddNodalSolutionStepVariable(SMA.TEMPERATURE_SENSITIVITY)

    fm = KratosForwardModel(model, fm_settings, entries)
    if fm.refs.size != 1:
        raise RuntimeError(f"expected one reference value, got {fm.refs.size}")

    sensor_file = fm_settings["sensor_data_file"].GetString()
    with open(sensor_file) as f:
        names = [s["name"] for s in json.load(f)["list_of_sensors"]]
    if len(names) != len(fm.located):
        raise RuntimeError(f"{sensor_file} lists {len(names)} sensors, "
                           f"the forward model located {len(fm.located)}")

    zone = entries[0]["target_sub_model_part"].GetString()
    return model, fm, primal, float(fm.refs[0]), sensor_file, names, zone


def primal_entity_names(primal):
    """Element and condition names the primal entity modeler creates."""
    for i in range(primal["modelers"].size()):
        p = primal["modelers"][i]["Parameters"]
        if p.Has("element_name") and p.Has("condition_name"):
            return p["element_name"].GetString(), p["condition_name"].GetString()
    raise RuntimeError("no modeler in the primal parameters names element_name and "
                       "condition_name")


# --------------------------------------------------------------------------
# Adjoint model
# --------------------------------------------------------------------------
def adjoint_constraints(primal, root_name):
    """The primal DISPLACEMENT / ROTATION fixities, moved to ADJOINT_DISPLACEMENT /
    ADJOINT_ROTATION on the same sub model parts and components. Adjoint fixities are
    homogeneous, so every constrained component gets 0. Anything else in the primal
    constraint list is refused rather than skipped."""
    procs = primal["processes"]["constraints_process_list"]
    out = []
    for i in range(procs.size()):
        p = procs[i]
        if p["process_name"].GetString() != "AssignVectorVariableProcess":
            raise RuntimeError(f"unexpected primal constraint process "
                               f"{p['process_name'].GetString()}")
        par = p["Parameters"]
        variable = par["variable_name"].GetString()
        if variable not in ("DISPLACEMENT", "ROTATION"):
            raise RuntimeError(f"unexpected primal constraint on {variable}")
        head, dot, tail = par["model_part_name"].GetString().partition(".")
        if head != root_name:
            raise RuntimeError(f"primal constraint on {head}{dot}{tail} is outside "
                               f"{root_name}")
        constrained = [par["constrained"][k].GetBool() for k in range(3)]
        out.append({
            "python_module": "assign_vector_variable_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorVariableProcess",
            "Parameters": {
                "model_part_name": ADJOINT_MODEL_PART + dot + tail,
                "variable_name": "ADJOINT_" + variable,
                "value": [0.0 if c else None for c in constrained],
                "constrained": constrained,
                "interval": json.loads(par["interval"].WriteJsonString()),
            },
        })
    return out


def adjoint_parameters(primal, sensitivity_part, constraints):
    """SystemIdentificationStaticAnalysis parameters, built here (no JSON file).
    No material_import_settings: the default materials_filename is "", so nothing is
    read, and the Properties come shared from the primal model part."""
    import KratosMultiphysics as Kratos

    solver = primal["solver_settings"]
    problem = primal["problem_data"]
    return Kratos.Parameters(json.dumps({
        "problem_data": {
            "problem_name": "adjoint_training_solves",
            "parallel_type": problem["parallel_type"].GetString(),
            "echo_level": 0,
            "start_time": problem["start_time"].GetDouble(),
            "end_time": problem["end_time"].GetDouble(),
        },
        "solver_settings": {
            "solver_type": "adjoint_static",
            "analysis_type": "linear",
            "model_part_name": ADJOINT_MODEL_PART,
            "domain_size": 3,
            "echo_level": 0,
            "time_stepping": json.loads(solver["time_stepping"].WriteJsonString()),
            # the adjoint linear strategy raises on either of these being true
            "compute_reactions": False,
            "move_mesh_flag": False,
            "rotation_dofs": solver["rotation_dofs"].GetBool(),
            "model_import_settings": {"input_type": "use_input_model_part"},
            "response_function_settings": {
                "perturbation_size": 1e-8,
                "adapt_perturbation_size": True,
            },
            "sensitivity_settings": {
                "sensitivity_model_part_name": sensitivity_part,
                "element_data_value_sensitivity_variables": ["YOUNG_MODULUS"],
                "build_mode": "static",
            },
            "linear_solver_settings":
                json.loads(solver["linear_solver_settings"].WriteJsonString()),
        },
        "processes": {
            "constraints_process_list": constraints,
            "loads_process_list": [],
            "list_other_processes": [],
        },
        "output_processes": {},
    }))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="SurrogateParameters.json")
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out = cfg["output"]
    e_scale = cfg["domain"]["e_scale_Pa"]

    # refuse to overwrite before any model is built
    out_path = os.path.join(out["dir"], OUTPUT_NAME)
    if not os.path.isdir(out["dir"]):
        sys.exit(f"output dir {os.path.abspath(out['dir'])} does not exist")
    if os.path.exists(out_path) and not args.overwrite:
        sys.exit(f"{out_path} exists; pass --overwrite to replace it")
    if not os.path.exists(out["training_csv"]):
        sys.exit(f"training CSV not found: {os.path.abspath(out['training_csv'])}")

    import KratosMultiphysics as Kratos
    from KratosMultiphysics.SystemIdentificationApplication.sensor_sensitivity_solvers \
        .system_identification_static_analysis import SystemIdentificationStaticAnalysis
    from KratosMultiphysics.SystemIdentificationApplication.utilities.sensor_utils \
        import CreateSensors

    print(f"Kratos {Kratos.Kernel.Version()}")

    # 1. primal model, adjoint nodal variables added before the mdpa is read
    model, fm, primal, e_ref, sensor_file, names, zone = build_primal(cfg["base_config"])
    n_sensors = len(names)
    primal_mp = fm.root
    root_name = primal_mp.Name
    print(f"primal model part {root_name}: {primal_mp.NumberOfNodes()} nodes, "
          f"{primal_mp.NumberOfElements()} elements, "
          f"{primal_mp.NumberOfConditions()} conditions; E_ref = {e_ref:.6e} Pa")

    # 5. training stiffnesses: existing CSV, sorted by E, equally spaced indices
    done = load_done(out["training_csv"], n_sensors)
    if not done:
        sys.exit(f"{out['training_csv']} holds no rows with status 'ok'")
    e_sorted = np.array(sorted(done))
    N = e_sorted.size
    if not 1 <= args.n_points <= N:
        sys.exit(f"--n-points must be between 1 and {N}")
    idx = np.unique(np.round(np.linspace(0, N - 1, args.n_points)).astype(int))
    if idx.size != args.n_points:
        sys.exit(f"--n-points {args.n_points} gives repeated indices out of {N}")
    e_points = e_sorted[idx]
    print(f"training CSV {out['training_csv']}: {N} solves; using indices "
          f"{idx.tolist()}")
    print(f"planned: {e_points.size} primal + {e_points.size * n_sensors} adjoint solves")

    adjoint = None
    try:
        # 2. adjoint model part sharing nodes, Properties, ProcessInfo and the
        #    variables list with the primal; elements and conditions are new objects
        #    with the primal ids, which the adjoint solver then replaces by their
        #    adjoint counterparts in this model part only
        element_name, condition_name = primal_entity_names(primal)
        adjoint_mp = model.CreateModelPart(ADJOINT_MODEL_PART)
        Kratos.ConnectivityPreserveModeler().GenerateModelPart(
            primal_mp, adjoint_mp, element_name, condition_name)
        print(f"{ADJOINT_MODEL_PART} generated with {element_name} / {condition_name}")

        # sensitivity model part: the E zone, which must hold every element
        head, _, sensitivity_part = zone.partition(".")
        if head != root_name or not adjoint_mp.HasSubModelPart(sensitivity_part):
            raise RuntimeError(f"zone {zone} has no counterpart in {ADJOINT_MODEL_PART}")
        n_zone = adjoint_mp.GetSubModelPart(sensitivity_part).NumberOfElements()
        if n_zone != adjoint_mp.NumberOfElements():
            raise RuntimeError(f"{sensitivity_part} holds {n_zone} of "
                               f"{adjoint_mp.NumberOfElements()} elements")

        # 3. adjoint analysis
        constraints = adjoint_constraints(primal, root_name)
        for c in constraints:
            p = c["Parameters"]
            print(f"  adjoint fixity {p['variable_name']:<22} on {p['model_part_name']:<36}"
                  f" constrained {p['constrained']}")
        adjoint = SystemIdentificationStaticAnalysis(
            model, adjoint_parameters(primal, sensitivity_part, constraints))
        adjoint.Initialize()

        # E changes between our solves, so the adjoint left-hand side must be rebuilt
        # on every solve
        strategy = adjoint._GetSolver()._GetSolutionStrategy()
        level = strategy.GetRebuildLevel()
        strategy.SetRebuildLevel(1)
        print(f"adjoint strategy rebuild level {level} -> {strategy.GetRebuildLevel()}")

        # 4. sensors on the adjoint model part, from the primal's sensor file
        with open(sensor_file) as f:
            sensor_root = Kratos.Parameters(f.read())
        sensor_list = sensor_root["list_of_sensors"]
        sensors = CreateSensors(model.CreateModelPart(SENSOR_MODEL_PART), adjoint_mp,
                                [sensor_list[i] for i in range(sensor_list.size())])
        if [s.GetName() for s in sensors] != names:
            raise RuntimeError(f"sensor order differs: {[s.GetName() for s in sensors]} "
                               f"vs {names}")
        for s in sensors:
            s.Initialize()

        sens_var = Kratos.KratosGlobals.GetVariable("YOUNG_MODULUS_SENSITIVITY")
        sens_mp = adjoint._GetSolver().GetSensitivityModelPart()
        responder = ExactResponse(fm, e_ref)
        n_adjoint = 0
        rows = []

        # 6. one primal and n_sensors adjoint solves per stiffness
        for e in e_points:
            # a. primal solve; u read as the existing code reads it, and the same
            #    sensors evaluated by the Kratos sensor objects
            u = responder(e)
            u_sensor = np.array([s.CalculateValue(primal_mp) for s in sensors])
            d_sensor = np.abs(u_sensor - u) / np.abs(u)

            # E per element as the adjoint elements see it; equal to the primal E only
            # if the Properties really are shared
            e_elem = np.array([el.Properties[Kratos.YOUNG_MODULUS] for el in sens_mp.Elements])
            if np.max(np.abs(e_elem - e)) > E_SHARED_TOL * e:
                raise RuntimeError(f"adjoint elements see E in [{e_elem.min():.6e}, "
                                   f"{e_elem.max():.6e}] Pa, primal E = {e:.6e} Pa")

            # b. one adjoint solve per sensor; Kratos stores minus the derivative
            #    du/dE_e, and dE_e/dt = E_e, so du/dt = -sum_e E_e * sensitivity_e
            dudt = np.empty(n_sensors)
            for i, s in enumerate(sensors):
                adjoint.CalculateGradient(s)
                n_adjoint += 1
                sens = np.array([el.GetValue(sens_var) for el in sens_mp.Elements])
                dudt[i] = -np.sum(e_elem * sens)
            ratio = dudt / (-u)

            # c. against the training CSV at the same E
            d_csv = np.abs(u - done[e]) / np.abs(done[e])

            print(f"\nE = {e / 1e9:8.3f} GPa  (t = {np.log(e / e_scale):+.6f})")
            print(f"  {'sensor':<16}{'u [m]':>16}{'du/dt [m]':>16}{'ratio':>20}"
                  f"{'|sensor-u|/|u|':>16}{'|u-csv|/|csv|':>16}")
            for i in range(n_sensors):
                print(f"  {names[i]:<16}{u[i]:>16.8e}{dudt[i]:>16.8e}{ratio[i]:>20.12f}"
                      f"{d_sensor[i]:>16.3e}{d_csv[i]:>16.3e}")

            # a negative ratio means the sign convention is not the one assumed above;
            # stop rather than flip it
            if np.any(ratio < 0.0):
                sys.exit(f"ratio is negative ({np.array2string(ratio, precision=12)}) at "
                         f"E = {e:.6e} Pa: the sensitivity sign is opposite to the assumed "
                         "convention. Stopping; nothing written.")

            ok = bool(np.all(np.abs(ratio - 1.0) <= RATIO_TOL))
            rows.append((e, u, dudt, ratio, "ok" if ok else "ratio_fail"))
    finally:
        if adjoint is not None:
            adjoint.Finalize()
        fm.Finalize()

    # 7. write the one output file
    cols = (["E_Pa"] + [f"u_{i}" for i in range(n_sensors)]
            + [f"dudt_{i}" for i in range(n_sensors)]
            + [f"ratio_{i}" for i in range(n_sensors)] + ["status"])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for e, u, dudt, ratio, status in rows:
            writer.writerow([repr(float(e))] + [repr(float(v)) for v in u]
                            + [repr(float(v)) for v in dudt]
                            + [repr(float(v)) for v in ratio] + [status])
    print(f"\nwritten -> {out_path}")

    # 8. version and solve counts
    print(f"Kratos {Kratos.Kernel.Version()}")
    print(f"primal solves: {responder.completed} (attempted {responder.attempted}, "
          f"forward model count {fm.n_solves}); adjoint solves: {n_adjoint}")

    failed = [r[0] for r in rows if r[4] != "ok"]
    if failed:
        sys.exit(f"ratio outside 1 +/- {RATIO_TOL:g} at E = "
                 + ", ".join(f"{e:.6e}" for e in failed) + " Pa (status ratio_fail)")


if __name__ == "__main__":
    main()
