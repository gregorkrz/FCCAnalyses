from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional

@dataclass
class Vec_RP:
    eta: list[float]
    phi: list[float]
    pt: list[float]
    txt: Optional[list] = None

class Event:
    def __init__(self, vec_rp: Vec_RP, vec_mc: Vec_RP = None, additional_collections={}):
        self.vec_rp = vec_rp
        self.vec_mc = vec_mc
        self.additional_collections = additional_collections # Special symbol to Vec_RP of other collections (e.g. GT quarks/gluons etc.)
        self.colors = []
    def display(self):
        fig, ax = plt.subplots()
        if self.vec_mc is not None:
            ax.scatter(self.vec_mc.eta, self.vec_mc.phi, s=self.vec_mc.pt, c='r', label='MC Particles', alpha=0.5)
        c = self.colors if self.colors else 'b'
        ax.scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.1)
        for label, vec in self.additional_collections.items():
            if vec.txt:
                for i, txt in enumerate(vec.txt):
                    ax.annotate(txt, (vec.eta[i], vec.phi[i]), fontsize=8, alpha=0.7)
            ax.scatter(vec.eta, vec.phi, s=vec.pt, label=label, alpha=0.5)
        # autoscale the axes
        #ax.autoscale()
        ax.set_xlim([-5, 5])
        ax.set_ylim([-3.5, 3.5])
        ax.set_xlabel('Eta')
        ax.set_ylabel('Phi')
        ax.set_title('Event Display')
        ax.legend()
        fig.tight_layout()
        return fig, ax
