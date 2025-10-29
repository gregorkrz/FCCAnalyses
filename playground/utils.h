
#include <cmath>
#include <vector>
#include <math.h>
#include "ROOT/RLogger.hxx"
#define rdfFatal R__LOG_FATAL(ROOT::Detail::RDF::RDFLogChannel())
#define rdfError R__LOG_ERROR(ROOT::Detail::RDF::RDFLogChannel())
#define rdfWarning R__LOG_WARNING(ROOT::Detail::RDF::RDFLogChannel())
#define rdfInfo R__LOG_INFO(ROOT::Detail::RDF::RDFLogChannel())
#define rdfDebug R__LOG_DEBUG(0, ROOT::Detail::RDF::RDFLogChannel())
#define rdfVerbose R__LOG_DEBUG(5, ROOT::Detail::RDF::RDFLogChannel())

#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/MCParticleData.h"
#include "edm4hep/ParticleIDData.h"
#include "ReconstructedParticle2MC.h"
#include <tuple>


namespace FCCAnalyses { namespace Utils {


    Vec_rp convert_calohits_to_vec_rp(const ROOT::VecOps::RVec<edm4hep::CalorimeterHitData> & calohits) {
        Vec_rp rp;
        for(auto & hit : calohits) {
            edm4hep::ReconstructedParticleData p;
            p.energy = hit.energy;
            p.mass = 0.;
            p.momentum.x = hit.position.x;
            p.momentum.y = hit.position.y;
            p.momentum.z = hit.position.z;
            p.PDG = 22; // photon
            rp.push_back(p);
        }
        return rp;
    }
    tuple<vector<float>, vector<float>, vector<float>, vector<int>> serialize_event(Vec_rp rp) {
    // return tuple of eta, phi, pt
    vector<float> eta;
    vector<float> phi;
    vector<float> pt;
    vector<int> pdg;
    vector<float> mass;
    for(auto & p : rp) {
        TLorentzVector p_lv;
        p_lv.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);
        eta.push_back(p_lv.Eta()); // Multiplied by 100 to convert
        phi.push_back(p_lv.Phi());
        pt.push_back(p_lv.Pt());
        pdg.push_back(p.PDG);
        mass.push_back(p.mass);
    }
    return tuple(eta, phi, pt, pdg, mass);
    }

    vector<float> rvec_to_vector(ROOT::VecOps::RVec<float> in) {
        vector<float> out;
        for(auto & v : in) {
            out.push_back(v);
        }
        return out;
    }

    int inspect_vecrp_object(Vec_rp object) {
        // Fill the vector with the mass of each item in the object
        rdfVerbose << "Inspecting the object" << endl;

        for(auto & j : object) {
            rdfVerbose <<  "object M = " << j.mass << endl;
        }
        return 0;
    }

}}


