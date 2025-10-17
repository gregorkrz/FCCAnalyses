
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
    tuple<vector<float>, vector<float>, vector<float>> serialize_event(Vec_rp rp) {
    // return tuple of eta, phi, pt
    vector<float> eta;
    vector<float> phi;
    vector<float> pt;
    for(auto & p : rp) {
        TLorentzVector p_lv;
        p_lv.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);
        eta.push_back(p_lv.Eta()); // Multiplied by 100 to convert
        phi.push_back(p_lv.Phi());
        pt.push_back(p_lv.Pt());
    }
    return tuple(eta, phi, pt);
    }
}}
