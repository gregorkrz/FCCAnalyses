
def get_Higgs_mass_with_truth_matching(df):
    df = df.Define("MC_quark_idx", "FCCAnalyses::ZHfunctions::get_MC_quark_index(Particle)")
    df = df.Define("MC_part_idx", "FCCAnalyses::ZHfunctions::get_MC_quark_index_for_Higgs(Particle, _Particle_daughters.index, false)")
    print("MC part. idx", df.AsNumpy(["MC_part_idx"])["MC_part_idx"])
    #df = df.Define("MC_part_asjets", "FCCAnalyses::ZHfunctions::get_jets_from_recojetlabels(MC_part_idx, FCCAnalyses::ZHfunctions::vec_mc_to_rp(Particle))")
    df = df.Define("MC_part_asjets", "FCCAnalyses::ZHfunctions::select_rp( FCCAnalyses::ZHfunctions::vec_mc_to_rp(Particle), MC_part_idx)")
    #print("MC_part_asjets", df.AsNumpy(["MC_part_asjets"])["MC_part_asjets"])
    df = df.Define("HardP_to_GenJet_mapping", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy(MC_part_asjets, GenJetDurhamN4, 0.2, false)")
    # For gen to reco jets, it should already be defined in fancy_matching
    #print("HardP_to_GenJet_mapping", df.AsNumpy(["HardP_to_GenJet_mapping"])["HardP_to_GenJet_mapping"])
    df = df.Define("HardP_to_RecoJet_mapping", "FCCAnalyses::ZHfunctions::merge_mappings(HardP_to_GenJet_mapping, fancy_matching)")
    # Select the given jets
    #print("HardP_to_RecoJet_mapping", df.AsNumpy(["HardP_to_RecoJet_mapping"])["HardP_to_RecoJet_mapping"])
    df = df.Define("filtered_jets", "FCCAnalyses::ZHfunctions::filter_jets(JetDurhamN4, HardP_to_RecoJet_mapping)")
    df = df.Define("filtered_jets_gen", "FCCAnalyses::ZHfunctions::filter_jets(GenJetDurhamN4, HardP_to_GenJet_mapping)")
    df = df.Define("inv_mass_reco", "FCCAnalyses::ZHfunctions::invariant_mass(filtered_jets)")
    #print("Inv mass reco", df.AsNumpy(["inv_mass_reco"])["inv_mass_reco"])
    df = df.Define("inv_mass_gen", "FCCAnalyses::ZHfunctions::invariant_mass(filtered_jets_gen)")
    #print("Inv mass gen", df.AsNumpy(["inv_mass_gen"])["inv_mass_gen"])
    return df
