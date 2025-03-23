from ovito.modifiers import *
from ovito.io import import_file
import numpy as np

# CONSTANTS

CORRECT_MAP_ICNA = {
    "al_fcc": CommonNeighborAnalysisModifier.Type.FCC,
    "fe_bcc": CommonNeighborAnalysisModifier.Type.BCC,
    "mg_hcp": CommonNeighborAnalysisModifier.Type.HCP,
}
CORRECT_MAP_CNA_NONDIAMOND = {
    "al_fcc": CommonNeighborAnalysisModifier.Type.FCC,
    "fe_bcc": CommonNeighborAnalysisModifier.Type.BCC,
    "mg_hcp": CommonNeighborAnalysisModifier.Type.HCP,
}
CORRECT_MAP_CNA_DIAMOND = {

}
CORRECT_MAP_ACKLAND_JONES = {
    "al_fcc": AcklandJonesModifier.Type.FCC,
    "fe_bcc": AcklandJonesModifier.Type.BCC,
    "mg_hcp": AcklandJonesModifier.Type.HCP,
}
CORRECT_MAP_VOROTOP = {

}

# HEURISTICS

def apply_heuristic(data_path, heuristic):
    pipeline = import_file(data_path)
    pipeline.modifiers.append(heuristic)

    num_frames = pipeline.source.num_frames
    all_data = []

    for frame in range(num_frames):
        data = pipeline.compute(frame)
        particle_info = data.particles_
        structures = particle_info["Structure Type"].__array__()
        all_data.append(structures)

    return np.concatenate(all_data) 


def compute_cna_nondiamond(data_path):
    cna = CommonNeighborAnalysisModifier()
    cna.structures[CommonNeighborAnalysisModifier.Type.ICO].enabled = False
    return apply_heuristic(data_path, cna)


def compute_icna(data_path):
    icna = CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.IntervalCutoff)
    icna.structures[CommonNeighborAnalysisModifier.Type.ICO].enabled = False
    return apply_heuristic(data_path, icna)


def compute_ackland_jones(data_path):
    ackland_jones = AcklandJonesModifier()
    ackland_jones.structures[CommonNeighborAnalysisModifier.Type.ICO].enabled = False
    return apply_heuristic(data_path, ackland_jones)


def compute_vorotop(data_path):
    vorotop = VoroTopModifier()
    vorotop.filter_file = "benchmarking/FCC-BCC-ICOS-both-HCP"
    return apply_heuristic(data_path, vorotop)


def compute_heuristic_accuracy(exp_name, data_path, heuristic):
    if heuristic == "Common Neighbor Analysis (Non-Diamond)":
        preds = compute_cna_nondiamond(data_path)
        return (preds == CORRECT_MAP_CNA_NONDIAMOND[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Interval Common Neighbor Alaysis":
        preds = compute_icna(data_path)
        return (preds == CORRECT_MAP_ICNA[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Ackland-Jones Analysis":
        preds = compute_ackland_jones(data_path)
        return (preds == CORRECT_MAP_ACKLAND_JONES[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "VoroTop Analysis":
        preds = compute_vorotop(data_path)
        return (preds == CORRECT_MAP_ACKLAND_JONES[exp_name]).sum().item() / len(preds)
    
    raise ValueError("Invalid heuristic name")