from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots



@dataclass
class Vec_RP:
    eta: list[float]
    phi: list[float]
    pt: list[float]
    txt: Optional[list] = None
    pdg: list[int] = None
    jets: list[int] = None

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

    def plot_eta_phi(self, title: str = "Event display: η–φ", show: bool = True):
        """
        Create an interactive η–φ plot for:
          - reconstructed particles (self.vec_rp)
          - generator-level particles (self.vec_mc, if present)
          - any additional collections (jets, etc.) in self.additional_collections

        Left subplot:
            - each collection has its own color
        Right subplot:
            - each collection has its own marker shape
            - color encodes jet index (from .jets, if present)
            - objects without a jet (or without .jets) are gray
        """
        # --- Create subplots: left = original, right = colored by jet ---
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[title, "η–φ colored by jet"],
            horizontal_spacing=0.08,
        )

        # Default color cycle (used on the left subplot for collections)
        default_colors = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#ff7f0e",  # orange
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#17becf",  # teal
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
        ]

        # Jet color palette for the right subplot (up to 6 jets)
        jet_palette = [
            "#1f77b4",  # jet 0
            "#d62728",  # jet 1
            "#2ca02c",  # jet 2
            "#ff7f0e",  # jet 3
            "#9467bd",  # jet 4
            "#8c564b",  # jet 5
        ]
        unassigned_color = "#7f7f7f"  # for -1 or missing jets
        jet_color_map = {}  # in case jet indices go beyond 0–5

        # Build list of collections to plot
        collections = []

        if self.vec_rp is not None:
            collections.append(("Reco", self.vec_rp))

        if self.vec_mc is not None:
            collections.append(("Gen", self.vec_mc))

        for label, vec in self.additional_collections.items():
            collections.append((label, vec))

        # Determine colors for collections (left subplot)
        n_cols = len(collections)
        if self.colors and len(self.colors) >= n_cols:
            color_list = self.colors[:n_cols]
        else:
            reps = (n_cols // len(default_colors)) + 1
            color_list = (default_colors * reps)[:n_cols]

        # Helper to build hover text
        def build_hover_text(label, vec, extra_text: Optional[list] = None):
            hover = []
            has_pdg = getattr(vec, "pdg", None) is not None and len(vec.pdg) == len(vec.eta)
            has_txt = getattr(vec, "txt", None) is not None and len(vec.txt) == len(vec.eta)
            has_extra = extra_text is not None and len(extra_text) == len(vec.eta)

            for i, (eta, phi, pt) in enumerate(zip(vec.eta, vec.phi, vec.pt)):
                parts = [
                    f"Collection: {label}",
                    f"η = {eta:.3f}",
                    f"φ = {phi:.3f}",
                    f"pT = {pt:.3f}",
                ]
                if has_pdg:
                    parts.append(f"PDG ID = {vec.pdg[i]}")
                if has_txt:
                    parts.append(f"txt = {vec.txt[i]}")
                if has_extra:
                    parts.append(str(extra_text[i]))
                hover.append("<br>".join(parts))
            return hover

        # Global pT scaling for marker sizes
        all_pts = []
        for _, vec in collections:
            all_pts.extend(vec.pt)

        max_pt = max(all_pts) if len(all_pts) > 0 else 1.0
        max_marker_size = 30
        min_marker_size = 6

        def pt_to_size(pt):
            return min_marker_size + (pt / max_pt) * (max_marker_size - min_marker_size)

        def get_jet_colors(vec):
            """
            Return a list of colors for each object in vec based on vec.jets.
            If .jets is missing / wrong length, default to gray.
            """
            jets = getattr(vec, "jets", None)
            n = len(vec.eta)

            # No jets information or length mismatch → all gray
            if jets is None or len(jets) != n:
                return [unassigned_color] * n

            colors = []
            for j in jets:
                # Unassigned / no jet
                if j is None or j < 0:
                    colors.append(unassigned_color)
                    continue

                # Map jet index → color
                if j not in jet_color_map:
                    # Assign a new color from jet_palette (cycle if needed)
                    idx = len(jet_color_map) % len(jet_palette)
                    jet_color_map[j] = jet_palette[idx]

                colors.append(jet_color_map[j])

            return colors

        # Add traces for each collection
        for idx, (label, vec) in enumerate(collections):
            if not vec or len(vec.eta) == 0 or len(vec.phi) == 0:
                continue

            extra_text = None
            if label in self.additional_text_dict:
                extra_text = self.additional_text_dict[label]

            hover = build_hover_text(label, vec, extra_text=extra_text)
            marker_symbol = self.special_symbols.get(label, "circle")
            sizes = [pt_to_size(pt) for pt in vec.pt]

            # --- Left subplot: colored by collection ---
            fig.add_trace(
                go.Scatter(
                    x=vec.eta,
                    y=vec.phi,
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=True,  # legend only from left subplot
                    text=hover,
                    hoverinfo="text",
                    marker=dict(
                        size=sizes,
                        color=color_list[idx],
                        symbol=marker_symbol,
                        line=dict(width=0.5),
                    ),
                ),
                row=1,
                col=1,
            )

            # --- Right subplot: colored by jet (or gray if no jet info) ---
            jet_colors = get_jet_colors(vec)

            fig.add_trace(
                go.Scatter(
                    x=vec.eta,
                    y=vec.phi,
                    mode="markers",
                    name=label,  # same name, but we hide from legend
                    legendgroup=label,
                    showlegend=False,
                    text=hover,  # you could extend this to include jet info if you like
                    hoverinfo="text",
                    marker=dict(
                        size=sizes,
                        color=jet_colors,
                        symbol=marker_symbol,  # shape still encodes collection
                        line=dict(width=0.5),
                    ),
                ),
                row=1,
                col=2,
            )

        # Axes ranges
        fig.update_xaxes(range=[-3.5, 3.5], title_text="η", row=1, col=1)
        fig.update_yaxes(range=[-3.5, 3.5], title_text="φ", row=1, col=1)

        fig.update_xaxes(range=[-3.5, 3.5], title_text="η", row=1, col=2)
        fig.update_yaxes(range=[-3.5, 3.5], title_text="φ", row=1, col=2)

        fig.update_layout(
            title=title,
            legend_title="Collections",
            template="plotly_white",
        )

        if show:
            fig.show()

        return fig

