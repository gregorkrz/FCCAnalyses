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
    def __init__(self, vec_rp: Vec_RP, vec_mc: Vec_RP = None, additional_collections={}, special_symbols={}, additional_text_dict={}):
        self.vec_rp = vec_rp
        self.vec_mc = vec_mc
        self.additional_collections = additional_collections # Special symbol to Vec_RP of other collections (e.g. GT quarks/gluons etc.)
        self.special_symbols = special_symbols # Maps an additional collection label to a special symbol for plotting
        self.colors = []
        self.additional_text_dict = additional_text_dict # Maps an additional collection label to a list of text annotations for that collection

    def display(self):
        fig, ax = plt.subplots(2, 4, figsize=(15, 12)) # First row: as before, second row: display  MC+InitialPartons | MC+GenJets | RP+RecoJets (using the current method) | RP+CaloJets
        if self.vec_mc is not None:
            ax[0, 0].scatter(self.vec_mc.eta, self.vec_mc.phi, s=self.vec_mc.pt, c='r', label='MC Particles', alpha=0.4, marker="o")
        c = self.colors if self.colors else 'b'
        # if additional text is not none, just print it on ax[0, 3] - it's not used anyway for plotting
        if self.additional_text_dict:
            text_str = ""
            for label, texts in self.additional_text_dict.items():
                text_str += "{}:\n".format(label)
                for t in texts:
                    text_str += "  {}\n".format(t)
            ax[0, 3].text(0.1, 0.5, text_str, fontsize=10, va='center', ha='left')
            ax[0, 3].axis('off')
        ax[0, 0].scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.4)
        ax[1, 0].scatter(self.vec_mc.eta, self.vec_mc.phi, s=self.vec_mc.pt, c='r', label='MC Particles', alpha=0.4, marker="o")
        ax[1, 1].scatter(self.vec_mc.eta, self.vec_mc.phi, s=self.vec_mc.pt, c='r', label='MC Particles', alpha=0.4, marker="o")
        ax[1, 0].set_title('MC Particles + Initial Partons')
        ax[1, 1].set_title('MC Particles + GenJets')
        ax[1, 2].set_title('Reconstructed Particles + PF Jets')
        ax[1, 3].set_title('Reconstructed Particles + Calo Jets')
        ax[1, 2].scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.4)
        ax[1, 3].scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.4)
        for label, vec in self.additional_collections.items():
            if vec.txt:
                for i, txt in enumerate(vec.txt):
                    for a in ax[0]:
                        a.annotate(txt, (vec.eta[i], vec.phi[i]), fontsize=8, alpha=0.7)
            for idx, a in enumerate(ax[0]):
                if idx == 2: continue
                a.scatter(vec.eta, vec.phi, s=vec.pt, label=label + " L={}".format(len(vec.pt)), alpha=0.5, marker=self.special_symbols.get(label, "o"))
        for key in ["RecoJets", "GenJets", "InitialPartons", "GenJetsFCCAnalysis", "CaloJets"]:
            markers = {
                "RecoJets": "*",
                "GenJets": "D",
                "InitialPartons": "X",
                "GenJetsFCCAnalysis": "P",
                "CaloJets": "H"
            }
            if key in self.additional_collections:
                # Plot RecoJets on ax[1,2], GenJets on ax[1,1], InitialPartons on ax[1,0], CaloJets on ax[1,3]. Also put the text with pt=,eta=,phi= for each collection
                plot_idx = {
                    "RecoJets": (1, 2),
                    "GenJets": (1, 1),
                    "InitialPartons": (1, 0),
                    "CaloJets": (1, 3)
                }
                if key in plot_idx:
                    i, j = plot_idx[key]
                    vec = self.additional_collections[key]
                    ax[i, j].scatter(vec.eta, vec.phi, s=np.array(vec.pt)*3.0, label=key + " L={}".format(len(vec.eta)), alpha=0.7, marker=markers[key])
                    # Also put text in place of each point with pt=,eta=,phi=
                    for n in range(len(vec.eta)):
                        ax[i, j].annotate("pt={:.2f}\neta={:.2f}\nphi={:.2f}".format(vec.pt[n], vec.eta[n], vec.phi[n]), (vec.eta[n], vec.phi[n]), fontsize=6, alpha=0.7)
                vec = self.additional_collections[key]
                ax[0, 2].scatter(vec.eta, vec.phi, s=np.array(vec.pt)*3.0, label=key + " L={}".format(len(vec.eta), str([round(x, 2) for x in vec.pt]), str([round(x, 2) for x in vec.eta])), alpha=0.7, marker=markers[key])
        # Autoscale the axes
        #ax.autoscale()
        for a in ax:
            for a1 in a:
                a1.set_xlim([-5, 5])
                #a.set_title('Event Display')
                a1.set_ylim([-3.5, 3.5])
                a1.set_xlabel('$\eta$')
                a1.set_ylabel('$\Phi$')
                a1.legend()
        fig.tight_layout()
        return fig, ax

