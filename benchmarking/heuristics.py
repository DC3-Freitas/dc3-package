from ovito.modifiers import *
from ovito.io import import_file
import numpy as np

# CONSTANTS (EXPECTED RESULTS)

CORRECT_MAP_ICNA = {
    "al_fcc": CommonNeighborAnalysisModifier.Type.FCC,
    "li_bcc": CommonNeighborAnalysisModifier.Type.BCC,
    "ti_hcp": CommonNeighborAnalysisModifier.Type.HCP
}
CORRECT_MAP_CNA_NONDIAMOND = {
    "al_fcc": CommonNeighborAnalysisModifier.Type.FCC,
    "li_bcc": CommonNeighborAnalysisModifier.Type.BCC,
    "ti_hcp": CommonNeighborAnalysisModifier.Type.HCP
}
CORRECT_MAP_CNA_DIAMOND = {
    "ge_cd": IdentifyDiamondModifier.Type.CUBIC_DIAMOND
}
CORRECT_MAP_ACKLAND_JONES = {
    "al_fcc": AcklandJonesModifier.Type.FCC,
    "li_bcc": AcklandJonesModifier.Type.BCC,
    "ti_hcp": AcklandJonesModifier.Type.HCP
}
CORRECT_MAP_VOROTOP = {
    "li_bcc": 2
}
CORRECT_MAP_CHILLPLUS = {
    "ge_cd": ChillPlusModifier.Type.CUBIC_ICE
}

# CONSTANTS (OTHER)

CHILLPLUS_CUTOFF = {
    "ge_cd": 2.9
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


def compute_cna_diamond(data_path):
    cna = IdentifyDiamondModifier()
    cna.structures[IdentifyDiamondModifier.Type.CUBIC_DIAMOND_FIRST_NEIGHBOR].enabled = False
    cna.structures[IdentifyDiamondModifier.Type.CUBIC_DIAMOND_SECOND_NEIGHBOR].enabled = False
    cna.structures[IdentifyDiamondModifier.Type.HEX_DIAMOND_FIRST_NEIGHBOR].enabled = False
    cna.structures[IdentifyDiamondModifier.Type.HEX_DIAMOND_SECOND_NEIGHBOR].enabled = False
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


def compute_chillplus(data_path, exp_name):
    chillplus = ChillPlusModifier()
    chillplus.cutoff = CHILLPLUS_CUTOFF[exp_name]
    return apply_heuristic(data_path, chillplus)


def compute_heuristic_accuracy(exp_name, data_path, heuristic):
    if heuristic == "Common Neighbor Analysis (Non-Diamond)":
        preds = compute_cna_nondiamond(data_path)
        return (preds == CORRECT_MAP_CNA_NONDIAMOND[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Common Neighbor Analysis (Diamond)":
        preds = compute_cna_diamond(data_path)
        return (preds == CORRECT_MAP_CNA_DIAMOND[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Interval Common Neighbor Alaysis":
        preds = compute_icna(data_path)
        return (preds == CORRECT_MAP_ICNA[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Ackland-Jones Analysis":
        preds = compute_ackland_jones(data_path)
        return (preds == CORRECT_MAP_ACKLAND_JONES[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "VoroTop Analysis":
        preds = compute_vorotop(data_path)
        return (preds == CORRECT_MAP_VOROTOP[exp_name]).sum().item() / len(preds)
    
    elif heuristic == "Chill+":
        preds = compute_chillplus(data_path, exp_name)
        return (preds == CORRECT_MAP_CHILLPLUS[exp_name]).sum().item() / len(preds)
    
    raise ValueError("Invalid heuristic name")