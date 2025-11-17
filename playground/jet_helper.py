def get_jet_vars(df, vec_rp_name, N_durham=-1, ee_pt_cutoff=-1, AK_radius=-1, name="FastJet_jets", is_ee_AK=False):
    '''
        vec_rp_name: name of the vector of ReconstructedParticles in the dataframe
    '''
    df = df.Define("rp_px_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_px({})".format(vec_rp_name))
    df = df.Define("rp_py_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_py({})".format(vec_rp_name))
    df = df.Define("rp_pz_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_pz({})".format(vec_rp_name))
    df = df.Define("rp_m_{}".format(name), "FCCAnalyses::ReconstructedParticle::get_mass({})".format(vec_rp_name))
    df = df.Define("fj_in_{}".format(name), "FCCAnalyses::JetClusteringUtils::set_pseudoJets_xyzm(rp_px_{},rp_py_{},rp_pz_{},rp_m_{})".format(name, name, name, name))
    if ee_pt_cutoff > 0:
        # We don't really use this
        df = df.Define(name, "JetClustering::clustering_ee_kt(0, {}, 1, 0)(fj_in_{})".format(ee_pt_cutoff, name))
    elif N_durham > 0:
        print("Using Durham jet clustering algorithm...")
        df = df.Define( name, "JetClustering::clustering_ee_kt(2, {}, 1, 0)(fj_in_{})".format(N_durham, name))
    else:
        assert AK_radius > 0
        if not is_ee_AK:
            print("Using AK with R=", AK_radius)
            df = df.Define( name, "JetClustering::clustering_antikt({}, 0, 0, 0, 0)(fj_in_{})".format( AK_radius, name))
        else:
            print("Using generalized e+e- AK with R=", AK_radius)
            df = df.Define( name, "JetClustering::clustering_ee_genkt({}, 0, 0, 0)(fj_in_{})".format( AK_radius, name))
        # For some reason, applying a pt cut here does not work and causes crashes
    # Define fj_eta, fj_phi, fj_pt variables # jets_durham is JetClustering::FCCAnalysesJet
    #df = df.Define("fj_eta", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_eta({}.jets))".format(name))
    #df = df.Define("fj_phi", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_phi({}.jets))".format(name))
    #df = df.Define("fj_pt", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_pt({}.jets))".format(name))
    return df

