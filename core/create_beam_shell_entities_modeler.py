import KratosMultiphysics as Kratos

# Phase-1 custom modeler for the beam.
#
# Beam.mdpa is geometry-based: Triangle2D3 / Line2D2 geometries, no Begin Elements.
# The core CreateEntitiesFromGeometriesModeler cannot build a 3D ShellThinElement3D3N from a
# 2D Triangle2D3 geometry, so entities are created directly here (CreateNewElement builds a
# flat z=0 Triangle3D3 from the node connectivity, which the shell element accepts).
#
# 3D shell elements are used so Phase-2's SystemIdentification adjoint solver (domain_size==3
# only) inverts against measurements produced by the same element formulation. If you only ever
# intend to run MCMC on forward solves, a 2D setup would also work -- see the notes.
#
# Elements are added back into each listed source SubModelPart. One shared Properties is
# created here; the materials file then assigns each listed SubModelPart its own Properties.


class CreateBeamShellEntitiesModeler(Kratos.Modeler):
    def __init__(self, model: Kratos.Model, settings: Kratos.Parameters):
        super().__init__(model, settings)
        self.model = model
        self.settings = settings
        self.settings.AddMissingParameters(Kratos.Parameters("""{
            "model_part_name"       : "Structure",
            "source_sub_model_parts" : ["Beam_Auto1"],
            "load_sub_model_part"   : "Load_on_lines_Auto1",
            "element_name"          : "ShellThinElement3D3N",
            "condition_name"        : "LineLoadCondition3D2N"
        }"""))

    def SetupModelPart(self) -> None:
        super().SetupModelPart()

        root = self.model[self.settings["model_part_name"].GetString()]
        properties = root.CreateNewProperties(1)

        element_name = self.settings["element_name"].GetString()
        condition_name = self.settings["condition_name"].GetString()

        n_elements = 0
        for name in self.settings["source_sub_model_parts"].GetStringArray():
            source = root.GetSubModelPart(name)
            element_ids = []
            for geometry in source.Geometries:
                node_ids = [node.Id for node in geometry]
                root.CreateNewElement(element_name, geometry.Id, node_ids, properties)
                element_ids.append(geometry.Id)
            source.AddElements(sorted(element_ids))
            n_elements += len(element_ids)

        load_group = root.GetSubModelPart(self.settings["load_sub_model_part"].GetString())
        condition_ids = []
        for geometry in load_group.Geometries:
            node_ids = [node.Id for node in geometry]
            root.CreateNewCondition(condition_name, geometry.Id, node_ids, properties)
            condition_ids.append(geometry.Id)
        load_group.AddConditions(sorted(condition_ids))

        Kratos.Logger.PrintInfo(
            "CreateBeamShellEntitiesModeler",
            "created %d elements, %d conditions" % (n_elements, len(condition_ids)))


def Factory(model: Kratos.Model, settings: Kratos.Parameters) -> CreateBeamShellEntitiesModeler:
    return CreateBeamShellEntitiesModeler(model, settings)