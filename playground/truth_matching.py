
def get_Higgs_mass_with_truth_matching(df, genjets_field="GenJetDurhamN4", recojets_field="JetDurhamN4"):
    #df = df.Define("MC_quark_idx", "FCCAnalyses::ZHfunctions::get_MC_quark_index(Particle)")
    df = df.Define("MC_part_idx", "FCCAnalyses::ZHfunctions::get_MC_quark_index_for_Higgs(Particle, _Particle_daughters.index, false)")
    print("MC part. idx", df.AsNumpy(["MC_part_idx"])["MC_part_idx"])
    df = df.Define("gt_labels", "FCCAnalyses::ZHfunctions::getGTLabels(MC_part_idx, Particle, _Particle_daughters.index);")
    # the nonzero indices of gt_labels should be filtered out
    df = df.Define("_gt_particles_from_higgs", "FCCAnalyses::ZHfunctions::select_gt_particles(gt_labels, Particle)")
    df = df.Define("gt_part_from_H_idx", "_gt_particles_from_higgs.first")
    df = df.Define("stable_gt_particles_from_higgs", "_gt_particles_from_higgs.second")
    df = df.Define("inv_mass_stable_gt_particles_from_higgs", "FCCAnalyses::ZHfunctions::invariant_mass(stable_gt_particles_from_higgs)")
    df = df.Define("reco_mc_links",
                   "FCCAnalyses::ZHfunctions::getRP2MC_index(_RecoMCLink_from.index, _RecoMCLink_to.index, ReconstructedParticles, Particle)")
    df = df.Define("mc2rp", "reco_mc_links.second")
    df = df.Define("reco_particles_matched_from_higgs", "FCCAnalyses::ZHfunctions::get_particles_from_mc2rp(gt_part_from_H_idx, mc2rp, ReconstructedParticles)")
    df = df.Define("inv_mass_reco_particles_matched_from_higgs", "FCCAnalyses::ZHfunctions::invariant_mass(reco_particles_matched_from_higgs)")
    #df = df.Define("MC_part_asjets", "FCCAnalyses::ZHfunctions::get_jets_from_recojetlabels(MC_part_idx, FCCAnalyses::ZHfunctions::vec_mc_to_rp(Particle))")
    df = df.Define("MC_part_asjets", "FCCAnalyses::ZHfunctions::select_rp( FCCAnalyses::ZHfunctions::vec_mc_to_rp(Particle), MC_part_idx)")
    df = df.Define("inv_mass_MC_part", "FCCAnalyses::ZHfunctions::invariant_mass(MC_part_asjets)")
    #print("MC_part_asjets", df.AsNumpy(["MC_part_asjets"])["MC_part_asjets"])
    df = df.Define("HardP_to_GenJet_mapping", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy(MC_part_asjets, {}, 1.0, false)".format(genjets_field))
    # For gen to reco jets, it should already be defined in fancy_matching
    #print("HardP_to_GenJet_mapping", df.AsNumpy(["HardP_to_GenJet_mapping"])["HardP_to_GenJet_mapping"])
    df = df.Define("HardP_to_RecoJet_mapping", "FCCAnalyses::ZHfunctions::merge_mappings(HardP_to_GenJet_mapping, fancy_matching)")
    # Select the given jets
    #print("HardP_to_RecoJet_mapping", df.AsNumpy(["HardP_to_RecoJet_mapping"])["HardP_to_RecoJet_mapping"])
    df = df.Define("filtered_jets", "FCCAnalyses::ZHfunctions::filter_jets({}, HardP_to_RecoJet_mapping)".format(recojets_field))
    df = df.Define("filtered_jets_gen", "FCCAnalyses::ZHfunctions::filter_jets({}, HardP_to_GenJet_mapping)".format(genjets_field))
    df = df.Define("inv_mass_reco", "FCCAnalyses::ZHfunctions::invariant_mass(filtered_jets)")
    #print("Inv mass reco", df.AsNumpy(["inv_mass_reco"])["inv_mass_reco"])
    df = df.Define("inv_mass_gen", "FCCAnalyses::ZHfunctions::invariant_mass(filtered_jets_gen)")
    df = df.Define("inv_mass_gen_all", "FCCAnalyses::ZHfunctions::invariant_mass({})".format(genjets_field))
    df = df.Define("inv_mass_reco_all", "FCCAnalyses::ZHfunctions::invariant_mass({})".format(recojets_field))
    #print("Inv mass gen", df.AsNumpy(["inv_mass_gen"])["inv_mass_gen"])
    return df

