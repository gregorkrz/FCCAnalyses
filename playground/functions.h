#ifndef ZHfunctions_H
#define ZHfunctions_H
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


namespace FCCAnalyses { namespace ZHfunctions {

// TRUTH TOOLS
vector<int> get_MC_quark_index(Vec_mc mc) { // Get the initial quarks from the MCParticle collection (the ones which jets we then detect)
    std::vector<int> quark_indices;
    for(size_t i = 0; i < mc.size(); ++i) {
        auto & p = mc[i];
        if (p.PDG < 5 && p.generatorStatus == 23) { // Quarks only (direct products of the hard interaction)
            quark_indices.push_back(i);
        }
    }
    //cout << "Quark indices: " << quark_indices << endl;
    // print the quark indices

    return quark_indices;
}




/*std::vector<int> get_list_of_particles_from_decay(int i,
                                             const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                                             const ROOT::VecOps::RVec<int>& ind) {
  std::vector<int> rest
  // i = index of a MC particle in the Particle block
  // in = the Particle collection
  // ind = the block with the indices for the daughters, Particle#1.index
  // returns a vector with the indices (in the Particle block) of the daughters of the particle i
  int db = in.at(i).daughters_begin;
  int de = in.at(i).daughters_end;
  if  ( db == de ) return res;   // particle is stable
  for (int id = db; id < de; id++) {
     res.push_back( ind[id] ) ;
  }
  return res;
}*/


pair<vector<int>,vector<int>>  getRP2MC_index(ROOT::VecOps::RVec<int> recind, ROOT::VecOps::RVec<int> mcind, Vec_rp reco, Vec_mc mc) {
  vector<int> result;
  vector<int> result_MC2RP;
  result.resize(reco.size(),-1.);
  result_MC2RP.resize(mc.size(),-1.);
  for (size_t i=0; i<recind.size();i++) {
    // if recind.at(i) is out of bounds, log a warning!
    if (recind.at(i) < 0 || recind.at(i) >= reco.size()) {
      //rdfVerbose << "getRP2MC_index: recind.at(" << i << ") = " << recind.at(i) << " is out of bounds [0," << reco.size()-1 << "]" << endl;
      continue;
    }
    result[recind.at(i)] = mcind.at(i);
  }
  for (size_t i=0; i<reco.size();i++) {
   if (result[i] <= -1 || result[i] >= mc.size()) {
      //rdfVerbose << "getRP2MC_index: result[" << i << "] = " << result[i] << " is out of bounds [-1," << mc.size()-1 << "]" << endl;
      continue;
    }
    if (result[i]>=0) {
        result_MC2RP[result[i]] = i;
    } else {
        result_MC2RP[result[i]] = -1;
    }
  }
  return make_pair(result, result_MC2RP);
}

string int_array_to_string(vector<int> arr) {
    string result = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        result += to_string(arr[i]);
        if (i < arr.size() - 1) result += ", ";
    }
    result += "]";
    return result;
}
std::vector<int> get_list_of_direct_particles_from_decay(int i,
                                             const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                                             const ROOT::VecOps::RVec<int>& ind) {

  std::vector<int> res;

  // i = index of a MC particle in the Particle block
  // in = the Particle collection
  // ind = the block with the indices for the daughters, Particle#1.index

  // returns a vector with the indices (in the Particle block) of the daughters of the particle i

  int db = in.at(i).daughters_begin ;
  int de = in.at(i).daughters_end;
  if  ( db == de ) return res;   // particle is stable
  for (int id = db; id < de; id++) {
     res.push_back( ind[id] ) ;
  }

  return res;
}
std::vector<int> get_list_of_end_decay_products_recursive(int i,
                                           const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                                           const ROOT::VecOps::RVec<int>& ind) {
   rdfVerbose << "  Particle idx " << i << "  direct daughters: " << int_array_to_string(get_list_of_direct_particles_from_decay(i, in, ind)) << endl;
   std::vector<int> res;
   int db = in.at(i).daughters_begin;
   int de = in.at(i).daughters_end;
  //rdfVerbose << "Particle idx " << i << " has daughters_begin = " << db << ", daughters_end = " << de << endl;
  // If no daughters, this particle is stable — return itself
  if (db == de) {
    res.push_back(i);
    return res;
  }
  // Otherwise, recurse through daughters
  for (int id = db; id < de; ++id) {
    int daughter_index = ind[id];
    auto daughters = get_list_of_end_decay_products_recursive(daughter_index, in, ind);
    res.insert(res.end(), daughters.begin(), daughters.end());
  }
  // convert res such that there are no duplicates
    std::sort(res.begin(), res.end());
    res.erase(std::unique(res.begin(), res.end()), res.end());
        //rdfVerbose  << " part_idx=" << i << ",decay_products=" << int_array_to_string(res) << endl;

    return res;
}

vector<float> min(vector<float> in) {
// Return a vector<float> of the minimum value in the input vector}
    vector<float> result;
    if(in.size() == 0) {
        result.push_back(-1);
        return result;
    }
    float min_val = in[0];
    for(auto & v : in) {
        if(v < min_val) min_val = v;
    }
    result.push_back(min_val);
    return result;
}

vector<float> get_jet_distances(Vec_rp jets) {
    // Get jet distances between each pair of jets (in deltaR)
    vector<float> result;
    for(auto & j : jets) {
        TLorentzVector j_lv;
        j_lv.SetXYZM(j.momentum.x, j.momentum.y, j.momentum.z, j.mass);
        for(auto & k : jets) {
            TLorentzVector k_lv;
            k_lv.SetXYZM(k.momentum.x, k.momentum.y, k.momentum.z, k.mass);
            float dR = j_lv.DeltaR(k_lv);
            if (dR > 0.0001) {
                // Make sure not to take the i-i pairs
                result.push_back(dR);
            }
        }
    }
    return result;
}

vector<int> getGTLabels(vector<int> initial_quarks, Vec_mc in, ROOT::VecOps::RVec<int> ind) {
    vector<int> result; // For each unique initial quark, get the list of all its decay products (recursively)
    // Set result to -1's of the size of in
    //rdfVerbose << "Getting GT labels for " << initial_quarks.size() << " initial quarks." << endl;
    result.resize(in.size(), -1);
    for (size_t i = 0; i < initial_quarks.size(); ++i) {
        vector<int> list_of_particles = get_list_of_end_decay_products_recursive(initial_quarks[i], in, ind);
        rdfVerbose << "Particle " << i << " (index " << initial_quarks[i] << ") has " << list_of_particles.size() << " decay products." << endl;
        // now print all of those decay products in one line. this needs to be done in a single rdfVerbose statement to avoid interleaving with other messages
        string decay_products = "";
        for (size_t j = 0; j < list_of_particles.size(); ++j) {
            decay_products += to_string(list_of_particles[j]) + " ";
        }
        rdfVerbose << "  Decay products: " << decay_products << endl;

        for (size_t j = 0; j < list_of_particles.size(); ++j) {
            if(result[list_of_particles[j]] == -1) { // Only set if not already set
                result[list_of_particles[j]] = i;
                //rdfVerbose << " GT label  " << i << " set to particle index " << list_of_particles[j] << endl;
            } /*else {
                rdfVerbose << " GT label  " << i << " NOT set to particle index " << list_of_particles[j] << " because it is already set to " << result[list_of_particles[j]] << endl;
            }*/
        }
    }
    // Exit, only proc one event
    exit(0);
    return result;
}


vector<int> convertMCJetLabelsIntoRecoJetLabels(vector<int> mc_labels, vector<int> mc2rp) {
    vector<int> result;
    for (size_t i = 0; i < mc_labels.size(); ++i) {
        if(mc_labels[i] >= 0 && mc_labels[i] < mc2rp.size()) {
            result.push_back(mc2rp[mc_labels[i]]);
        }
        else {
            result.push_back(-1);
        }
    }
    return result;
}
struct rp {
float momentum[3] = {0,0,0};
    float energy = 0;
    float mass = 0;
    int charge = 0;
};

Vec_rp convert(vector<rp> in) {
    Vec_rp out;
     //vec_rp is ROOT:VecOps::RVec< edm4hep::ReconstructedParticleData >
    for (auto & p : in) {
        edm4hep::ReconstructedParticleData rp;
        rp.momentum.x = p.momentum[0];
        rp.momentum.y = p.momentum[1];
        rp.momentum.z = p.momentum[2];
        rp.energy = p.energy;
        rp.mass = p.mass;
        rp.charge = p.charge;
        out.push_back(rp);
    }
    return out;
}


Vec_rp get_GT_jets_from_initial_particles(Vec_mc mc_particles, vector<int> quark_idx) {
    // Picks the intial quarks and returns them in the same format as the jets, so that they can be used in the existing matching functions
    vector<rp> result;
    for (auto & i : quark_idx) {
        if(i >= 0 && i < mc_particles.size()) {
            rp p;
            p.momentum[0] = mc_particles[i].momentum.x;
            p.momentum[1] = mc_particles[i].momentum.y;
            p.momentum[2] = mc_particles[i].momentum.z;
            //p.energy = mc_particles[i].energy;
            // get energy
            p.energy = std::sqrt(mc_particles[i].momentum.x * mc_particles[i].momentum.x +
                                     mc_particles[i].momentum.y * mc_particles[i].momentum.y +
                                     mc_particles[i].momentum.z * mc_particles[i].momentum.z +
                                     mc_particles[i].mass * mc_particles[i].mass);
            const float px = p.momentum[0];
            const float py = p.momentum[1];
            const float pz = p.momentum[2];
            const float e  = p.energy;
            const float m2 = e*e - (px*px + py*py + pz*pz);
            p.mass = (m2 > 0.f) ? std::sqrt(m2) : 0.f;
            p.charge = 0; // Quarks have fractional charge, set to 0 for jets
            result.push_back(p);
        }
    }
    return convert(result);
}

Vec_rp get_jets_from_recojetlabels(vector<int> RecoJetLabels, Vec_rp RecoParticles) {
  vector<rp> result;
  // Basic sanity check
  if (RecoJetLabels.size() != RecoParticles.size()) return convert(result);
  // Find the largest non-negative label (i.e. number of jets - 1)
  int maxLabel = -1;
  for (int l : RecoJetLabels) if (l > maxLabel) maxLabel = l;
  if (maxLabel < 0) return convert(result); // nothing to cluster
  // allocate jets [0..maxLabel]
  result.resize(static_cast<size_t>(maxLabel + 1));
  for (auto& j : result) {
    j.momentum[0] = 0.f;
    j.momentum[1] = 0.f;
    j.momentum[2] = 0.f;
    j.energy      = 0.f;
    j.mass        = 0.f;
    j.charge      = 0;
  }
  for (size_t i = 0; i < RecoJetLabels.size(); ++i) {
    int l = RecoJetLabels[i];
    if (l < 0) continue; // -1: Do not cluster
    const auto& p = RecoParticles[i];
    auto&       j = result[static_cast<size_t>(l)];
    j.momentum[0] += p.momentum[0];
    j.momentum[1] += p.momentum[1];
    j.momentum[2] += p.momentum[2];
    j.energy      += p.energy;
    j.charge      += p.charge; // integer sum is fine in edm4hep
  }
  for (auto& j : result) {
    const float px = j.momentum[0];
    const float py = j.momentum[1];
    const float pz = j.momentum[2];
    const float e  = j.energy;
    const float m2 = e*e - (px*px + py*py + pz*pz);
    j.mass = (m2 > 0.f) ? std::sqrt(m2) : 0.f;
  }
  return convert(result);
}

/*std::vector<int> get_list_of_particles_from_decay(int i, vector<edm4hep::MCParticleData> in, vector<int> ind) {

  std::vector<int> res;
  // i = index of a MC particle in the Particle block
  // in = the Particle collection
  // ind = the block with the indices for the daughters, Particle#1.index
  // returns a vector with the indices (in the Particle block) of the daughters of the particle i

  int db = in.at(i).daughters_begin ;
  int de = in.at(i).daughters_end;
  if  ( db == de ) return res;   // particle is stable
  for (int id = db; id < de; id++) {
     res.push_back( ind[id] ) ;
  }
  return res;
}
*/

vector<float> get_jet_eta(Vec_rp jets) {
    std::vector<float> eta;
    for(auto & j : jets) {
        TLorentzVector j_lv;
        j_lv.SetXYZM(j.momentum.x, j.momentum.y, j.momentum.z, j.mass);
        eta.push_back(j_lv.Eta());
    }
    return eta;
}

float max_jet_energy(Vec_rp jets) {
    float max_e = 0;
    for(auto & j : jets) {
        if(j.energy > max_e) max_e = j.energy;
    }
    return max_e;
}

std::vector<float> sort_jet_energies(Vec_rp jets) { // return a vector<float> of the jet energies highest to lowest
    std::vector<float> energies;
    for(auto & j : jets) {
        energies.push_back(j.energy);
    }
    std::sort(energies.begin(), energies.end(), std::greater<float>());
    return energies;
}

std::vector<int> get_reco_truth_jet_mapping_greedy(Vec_rp reco_jets, Vec_rp gen_jets, float dR)
{
    // match reco jets to gen jets one-to-one by smallest ΔR, return vector<int> with
    // index of matched gen jet for each reco jet, -1 if no match found
    std::vector<int> result(reco_jets.size(), -1);
    std::vector<char> used(gen_jets.size(), 0);
    struct Pair { int i, j; float dR2; };
    std::vector<Pair> pairs;
    pairs.reserve(reco_jets.size() * gen_jets.size());
    // Precompute ΔR² for all pairs within cone
    for (size_t i = 0; i < reco_jets.size(); ++i) {
        TLorentzVector rj_lv;
        rj_lv.SetXYZM(reco_jets[i].momentum.x,
                      reco_jets[i].momentum.y,
                      reco_jets[i].momentum.z,
                      reco_jets[i].mass);
        for (size_t j = 0; j < gen_jets.size(); ++j) {
            TLorentzVector gj_lv;
            gj_lv.SetXYZM(gen_jets[j].momentum.x,
                          gen_jets[j].momentum.y,
                          gen_jets[j].momentum.z,
                          gen_jets[j].mass);
            float dRval = rj_lv.DeltaR(gj_lv);
            if (dRval < dR)
                pairs.push_back({(int)i, (int)j, dRval * dRval});
        }
    }
    // Sort by smallest ΔR first
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair &a, const Pair &b) { return a.dR2 < b.dR2; });
    // Greedy one-to-one assignment
    for (auto &p : pairs) {
        if (result[p.i] == -1 && !used[p.j]) {
            result[p.i] = p.j;
            used[p.j] = 1;
        }
    }
    return result;
}


std::vector<int> get_reco_truth_jet_mapping(Vec_rp reco_jets, Vec_rp gen_jets, float dR) {
// match reco jets to gen jets, return a vector<int> with the index of the matched gen jet for each reco jet, -1 if no match found
    std::vector<int> result;
    for(auto & rj : reco_jets) {
        int idx = -1;
        float dR_min = dR; // Matching cone
        TLorentzVector rj_lv;
        rj_lv.SetXYZM(rj.momentum.x, rj.momentum.y, rj.momentum.z, rj.mass);
        for(size_t i = 0; i < gen_jets.size(); ++i) {
            auto & gj = gen_jets[i];
            TLorentzVector gj_lv;
            gj_lv.SetXYZM(gj.momentum.x, gj.momentum.y, gj.momentum.z, gj.mass);
            float dR = rj_lv.DeltaR(gj_lv);
            if(dR < dR_min) {
                dR_min = dR;
                idx = i;
            }
        }
        result.push_back(idx);
    }
    return result;
}

vector<float> filter_number_by_bin(vector<float> values, vector<float> binning, float lower_bound, float upper_bound) { // Return a vector<float> of the values that fall within the specified bin range
    std::vector<float> result;
    // values and binning have the same size. go through binning and, if the current value is within the specified range, add it to the result
    if(values.size() != binning.size()) {
        std::cout << "ERROR: filter_number_by_bin, values and binning must be of same size." << values.size() << "  " << binning.size() << std::endl;
        exit(1);
    }
    //cout << "-----" << endl;
    //return result;
    for (size_t i = 0; i < values.size(); ++i) {
        //cout << "binning[" << i << "] = " << binning[i] << ", values[" << i << "] = " << values[i] << std::endl;
        if (binning[i] >= lower_bound && binning[i] < upper_bound && values[i] != -1) {
            result.push_back(values[i]);
        }
    }
    return result;
}

tuple<vector<float>, vector<float>> matched_genjet_E_and_all_genjet_E(vector<int> reco_to_gen_matching, Vec_rp gen_jets) {
    // Return a pair of vector<float>, first is the energies of matched gen jets, second is the energies of unmatched gen jets
    std::vector<float> matched;
    std::vector<float> all_genjet;
    std::vector<char> used(gen_jets.size(), 0);
    for(size_t i = 0; i < reco_to_gen_matching.size(); ++i) {
        int idx = reco_to_gen_matching[i];
        if(idx >= 0 && idx < gen_jets.size()) {
            matched.push_back(gen_jets[idx].energy);
            all_genjet.push_back(gen_jets[idx].energy);
            used[idx] = 1;
        }
    }
    for(size_t i = 0; i < gen_jets.size(); ++i) {
        if(!used[i]) {
            all_genjet.push_back(gen_jets[i].energy);
        }
    }
    return tuple(matched, all_genjet);
}

tuple<vector<float>, vector<float>, vector<float>, vector<float>> get_energy_ratios_for_matched_jets(vector<int> reco_to_gen_matching, Vec_rp reco_jets, Vec_rp gen_jets) {
    // Return a vector<float> of the energy ratios of matched jets
    std::vector<float> result;
    std::vector<float> recojetE;
    vector<float> unmatched_reco_jet_E;
    vector<float> genjetE;
    vector<float> genjetEta;
    for(size_t i = 0; i < reco_to_gen_matching.size(); ++i) {
        int idx = reco_to_gen_matching[i];
        if(idx >= 0 && idx < gen_jets.size()) {
            float ratio = reco_jets[i].energy / gen_jets[idx].energy;
            result.push_back(ratio);
            recojetE.push_back(reco_jets[i].energy);
            genjetE.push_back(gen_jets[idx].energy);
            TLorentzVector gj_lv;
            gj_lv.SetXYZM(gen_jets[idx].momentum.x, gen_jets[idx].momentum.y, gen_jets[idx].momentum.z, gen_jets[idx].mass);
            genjetEta.push_back(gj_lv.Eta());

        }
        else {
            result.push_back(-1);
            recojetE.push_back(reco_jets[i].energy);
            genjetE.push_back(-1);
            genjetEta.push_back(-100);
            unmatched_reco_jet_E.push_back(reco_jets[i].energy);
        }

    }
    std::vector<size_t> indices(result.size());
    for(size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    //std::sort(indices.begin(), indices.end(), [&result](size_t a, size_t b) { return result[2*a+1] > result[2*b+1]; });
    sort(indices.begin(), indices.end(), [&recojetE](size_t a, size_t b) { return recojetE[a] > recojetE[b]; });
    std::vector<float> sorted_result;
    vector<float> sorted_genjetE;
    vector<float> sorted_genjetEta;
    for(size_t i = 0; i < indices.size(); ++i) {
        sorted_result.push_back(result[indices[i]]);
        sorted_genjetE.push_back(genjetE[indices[i]]);
        sorted_genjetEta.push_back(genjetEta[indices[i]]);
    }
    return tuple(sorted_result, unmatched_reco_jet_E, sorted_genjetE, sorted_genjetEta);
}

std::vector<float> elementwise_divide(vector<float> v1, vector<float> v2) { // return a vector<float> of the element-wise division of two vectors
    std::vector<float> result;
    if(v1.size() != v2.size()) {
        std::cout << "ERROR: elementwise_divide, vectors must be of same size." << std::endl;
        exit(1);
    }
    for(size_t i = 0; i < v1.size(); ++i) {
        if(v2[i] == 0) {
            result.push_back(0);
        }
        else {
            result.push_back(v1[i]/v2[i]);
        }
    }
    return result;
}
}}
#endif
