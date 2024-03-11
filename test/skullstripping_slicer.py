import os
from glob import glob
from pathlib import Path, PurePath, PureWindowsPath

os.chdir("S:\\StrokeAI\\healthy_patients")

def swissSkullStrip(input_path, output_path, output_mask_path, atlas_path, atlas_mask_path):
    slicer.mrmlScene.Clear(0)
    sss = slicer.modules.swissskullstripper
    parameters = {}
    parameters["atlasMRIVolume"] = slicer.util.loadVolume(atlas_path)
    parameters["atlasMaskVolume"] = slicer.util.loadVolume(atlas_mask_path, {"labelmap":True})
    parameters["patientVolume"] = slicer.util.loadVolume(input_path)
    outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputMaskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    parameters["patientOutputVolume"] = outputNode
    parameters["patientMaskLabel"] = outputMaskNode
    run = slicer.cli.runSync(sss, None, parameters)
    return slicer.util.saveNode(outputMaskNode, output_mask_path), slicer.util.saveNode(outputNode, output_path)

def getPaths(input_path):
    path_obj = PureWindowsPath(input_path)
    pat_name = path_obj.parent
    output_path = pat_name.joinpath(str(pat_name) + "_skullstrip.nii")
    output_mask_path = pat_name.joinpath(str(pat_name) + "_skullstrip_mask.nii")
    atlas_path = "S:\\StrokeAI\\atlasImage.mha"
    atlas_mask_path = "S:\StrokeAI\Try\co-regis\SkullStrip\\atlasmask-label.mha"
    return input_path, "S:\\StrokeAI\\healthy_patients\\" + str(output_path), "S:\\StrokeAI\\healthy_patients\\" + str(output_mask_path), atlas_path, atlas_mask_path

paths = sorted(glob("*\*final*"))
for pat_path in paths:

    input_path, output_path, output_mask_path, atlas_path, atlas_mask_path = getPaths(pat_path)
    print(pat_path)
    print(input_path)
    print(output_path)
    try:
        swissSkullStrip(input_path, output_path, output_mask_path, atlas_path, atlas_mask_path)
    except:
        print(pat_path, "failed")
        continue
    break


