from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Vec_RP:
    eta: list[float]
    phi: list[float]
    pt: list[float]
    txt: list

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
        ax.scatter(self.vec_rp.eta, self.vec_rp.phi, s=self.vec_rp.pt, c=c, label='Reconstructed Particles', alpha=0.5)
        if self.vec_rp.txt:
            # also "plot" the text onto the plots
            for i, txt in enumerate(self.vec_rp.txt):
                ax.annotate(txt, (self.vec_rp.eta[i], self.vec_rp.phi[i]), fontsize=8, alpha=0.7)
        for label, vec in self.additional_collections.items():
            ax.scatter(vec.eta, vec.phi, s=vec.pt, label=label, alpha=0.5)
        ax.set_xlabel('Eta')
        ax.set_ylabel('Phi')
        ax.set_title('Event Display')
        ax.legend()
        fig.tight_layout()
        return fig, ax

