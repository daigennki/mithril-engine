bl_info = {
    "name":         "MMAT Exporter",
    "version":      (2, 82, 0),
    "blender":      (2, 82, 0),
    "author":       "daigennki",
    "description":  "Generate MithrilEngine Material (.mmat) file from a material's node tree",
    "category":     "Import-Export"
}

import os
import errno
import bpy
import json
import pathlib

class MMAT_ExportPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_label = "MithrilEngine Material"

    def draw(self, context):
        self.layout.operator("mmat.export", text="Export")
        self.layout.operator("mmat.batch", text="Export all in scene")

def simplifyPath(path_in):
    path_out = path_in
    path_out = path_out.replace('\\', '/')
    path_out = path_out.replace("//../materials/", "")  # erase relative directory if this Blender file is in the engine directory (for dev purposes)
    return path_out

def exportMaterial(use_mat):
    # Generate MMAT file
    mmat_data = {}

    # Get the shader connected to this material's input in the node tree
    mat_output = use_mat.node_tree.nodes["Material Output"]
    if not mat_output.inputs["Surface"].is_linked:
        raise TypeError("Shader not linked to 'Surface' input of 'Material Output'!")
    ms = mat_output.inputs["Surface"].links[0].from_node

    # Get data from shader inputs
    ## Diffuse map ('Base Color' or 'Color')
    if "Base Color" in ms.inputs:
        mat_basecolor = ms.inputs["Base Color"]
    else:
        mat_basecolor = ms.inputs["Color"]
    if mat_basecolor.is_linked:
        mmat_data["Diffuse"] = simplifyPath(mat_basecolor.links[0].from_node.image.filepath)
    else:
        # Get RGBA color if no texture is linked
        basecolor_array = [ 1.0, 1.0, 1.0, 1.0 ]
        basecolor_array[0] = mat_basecolor.default_value[0]
        basecolor_array[1] = mat_basecolor.default_value[1]
        basecolor_array[2] = mat_basecolor.default_value[2]
        basecolor_array[3] = mat_basecolor.default_value[3]
        mmat_data["Diffuse"] = basecolor_array
    ## Normal map
    if "Normal" in ms.inputs:
        mat_normal = ms.inputs["Normal"]
        if mat_normal.is_linked:
            mmat_data["Normal"] = simplifyPath(mat_normal.links[0].from_node.image.filepath)
    ## Specular map 
    if "Specular" in ms.inputs:
        mat_specular = ms.inputs["Specular"]
        if mat_specular.is_linked:
            mmat_data["Specular"] = simplifyPath(mat_specular.links[0].from_node.image.filepath)
        else:
            mmat_data["Specular"] = mat_specular.default_value
    ## Translucent
    if "Alpha" in ms.inputs:
        mat_alpha = ms.inputs["Alpha"]
        if mat_alpha.is_linked:
            mmat_data["Translucent"] = True

    # Dump material data into MMAT file as JSON data
    filename = bpy.path.abspath("//" + use_mat.name + ".mmat")
    # Create directory if necessary
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    mmat_file = open(filename, "w+")
    json.dump(mmat_data, mmat_file, ensure_ascii=False, indent=4)
    mmat_file.close()

    return filename

class MMAT_ExportButton(bpy.types.Operator):
    bl_idname = "mmat.export"
    bl_label = "text"

    def execute(self, context):
        # Generate single MMAT file from active material
        use_mat = bpy.context.object.active_material
        self.report({ "INFO" }, "Exported " + exportMaterial(use_mat))
        return { "FINISHED" }

class MMAT_BatchExportButton(bpy.types.Operator):
    bl_idname = "mmat.batch"
    bl_label = "text"

    def execute(self, context):
        # Generate multiple MMAT files from all materials in scene
        already_exported = []
        exported = 0
        for obj in bpy.context.scene.objects:
            for mat_slot in obj.material_slots:
                if mat_slot:
                    if mat_slot.material not in already_exported:
                        exportMaterial(mat_slot.material)
                        already_exported.append(mat_slot.material)
                        exported += 1
        self.report({ "INFO" }, "Exported " + str(exported) + " MMAT files")
        return { "FINISHED" }

classes = [
    MMAT_ExportPanel,
    MMAT_ExportButton,
    MMAT_BatchExportButton
]
def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()