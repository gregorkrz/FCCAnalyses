


def get_jet_vars(df, vec_rp_name, N_durham=4):
    '''
    vec_rp_name: name of the vector of ReconstructedParticles in the dataframe
    '''
    df = df.Define("rp_px", "FCCAnalyses::ReconstructedParticle::get_px({})".format(vec_rp_name))
    df = df.Define("rp_py", "FCCAnalyses::ReconstructedParticle::get_py({})".format(vec_rp_name))
    df = df.Define("rp_pz", "FCCAnalyses::ReconstructedParticle::get_pz({})".format(vec_rp_name))
    df = df.Define("rp_e", "FCCAnalyses::ReconstructedParticle::get_p({})".format(vec_rp_name))
    df = df.Define("fj_in", "FCCAnalyses::JetClusteringUtils::set_pseudoJets(rp_px,rp_py,rp_pz,rp_e)")
    df = df.Define( "jets_durham", "JetClustering::clustering_ee_kt(2, {}, 1, 0)(fj_in)".format(N_durham))
    # Define fj_eta, fj_phi, fj_pt variables # jets_durham is JetClustering::FCCAnalysesJet
    df = df.Define("fj_eta", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_eta(jets_durham.jets))")
    df = df.Define("fj_phi", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_phi(jets_durham.jets))")
    df = df.Define("fj_pt", "FCCAnalyses::Utils::rvec_to_vector(FCCAnalyses::JetClusteringUtils::get_pt(jets_durham.jets))")
    
    return df

