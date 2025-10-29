from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

@dataclass
class Vec_RP:
    eta: list[float]
    phi: list[float]
    pt: list[float]
    txt: Optional[list] = None

class Event:
    def __init__(self, vec_rp: Vec_RP, vec_mc: Vec_RP = None, additional_collections={}, special_symbols={}):
        self.vec_rp = vec_rp
        self.vec_mc = vec_mc
        self.additional_collections = additional_collections # Special symbol to Vec_RP of other collections (e.g. GT quarks/gluons etc.)
        self.special_symbols = special_symbols # Maps an additional collection label to a special symbol for plotting
        self.colors = []
    def display(self):
        fig, ax = plt.subplots(1, 3, figsize=(13, 5))
        if self.vec_mc is not None:
            ax[0].scatter(self.vec_mc.eta, self.vec_mc.phi, s=self.vec_mc.pt, c='r', label='MC Particles', alpha=0.4, marker="o")
        c = self.colors if self.colors else 'b'
        ax[0].scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.4)
        for label, vec in self.additional_collections.items():
            if vec.txt:
                for i, txt in enumerate(vec.txt):
                    for a in ax:
                        a.annotate(txt, (vec.eta[i], vec.phi[i]), fontsize=8, alpha=0.7)
            for idx, a in enumerate(ax):
                if idx == 2: continue
                a.scatter(vec.eta, vec.phi, s=vec.pt, label=label + " L={}".format(len(vec.pt)), alpha=0.5, marker=self.special_symbols.get(label, "o"))
        for key in ["RecoJets", "GenJets", "InitialPartons", "GenJetsFCCAnalysis"]:
            markers = {
                "RecoJets": "*",
                "GenJets": "D",
                "InitialPartons": "X",
                "GenJetsFCCAnalysis": "P"
            }
            if key in self.additional_collections:
                if key in self.additional_collections:
                    vec = self.additional_collections[key]
                    ax[2].scatter(vec.eta, vec.phi, s=np.array(vec.pt)*3.0, label=key + " L={}".format(len(vec.eta), str([round(x, 2) for x in vec.pt]), str([round(x, 2) for x in vec.eta])), alpha=0.7, marker=markers[key])

        # Autoscale the axes
        #ax.autoscale()
        for a in ax:
            a.set_xlim([-5, 5])
            #a.set_title('Event Display')
            a.set_ylim([-3.5, 3.5])
            a.set_xlabel('Eta')
            a.set_ylabel('Phi')
            a.legend()
        fig.tight_layout()
        return fig, ax

