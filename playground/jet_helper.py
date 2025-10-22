


def get_jet_vars(df, vec_rp_name, N_durham=4, ee_pt_cutoff=-1, name="FastJet_jets"):
    '''
        vec_rp_name: name of the vector of ReconstructedParticles in the dataframe
    '''
    df = df.Define("rp_px_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_px({})".format(vec_rp_name))
    df = df.Define("rp_py_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_py({})".format(vec_rp_name))
    df = df.Define("rp_pz_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_pz({})".format(vec_rp_name))
    df = df.Define("rp_e_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_p({})".format(vec_rp_name))
    df = df.Define("fj_in_{}".format(name), "FCCAnalyses::JetClusteringUtils::set_pseudoJets(rp_px_{},rp_py_{},rp_pz_{},rp_e_{})".format(name, name, name, name))
    if ee_pt_cutoff > 0:
        # jets_durham is not a good name here, but keep it for consistency
        df = df.Define(name, "JetClustering::clustering_ee_kt(0, {}, 1, 0)(fj_in_{})".format(ee_pt_cutoff, name))
    else:
        df = df.Define( name, "JetClustering::clustering_ee_kt(2, {}, 1, 0)(fj_in_{})".format(N_durham, name))
    # Define fj_eta, fj_phi, fj_pt variables # jets_durham is JetClustering::FCCAnalysesJet
    #df = df.Define("fj_eta", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_eta({}.jets))".format(name))
    #df = df.Define("fj_phi", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_phi({}.jets))".format(name))
    #df = df.Define("fj_pt", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_pt({}.jets))".format(name))

    return df


