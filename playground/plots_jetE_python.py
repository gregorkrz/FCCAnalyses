import ROOT

# Open the ROOT file
f = ROOT.TFile.Open("../../idea_fullsim/fast_sim/histograms/p8_ee_WW_ecm365_fullhad.root")

# Get histograms
hists = [
    (f.Get("h_E0"), "1st"),
    (f.Get("h_E1"), "2nd"),
    (f.Get("h_E2"), "3rd"),
    (f.Get("h_E3"), "4th")
]

# Set up canvas
c = ROOT.TCanvas("c", "Histograms", 800, 600)

# Define colors/markers for distinction
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]

legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)

# Loop through histograms, style them, and draw
for i, (h, label) in enumerate(hists):
    if not h:
        print(f"Warning: histogram {label} not found in file")
        continue
    h.SetLineColor(colors[i])
    h.SetLineWidth(2)
    h.SetStats(0)
    h.GetXaxis().SetTitle("E_{reco}/E_{true}")
    h.GetYaxis().SetTitle("Events")
    #logscale
    c.SetLogy()
    drawopt = "HIST" if i == 0 else "HIST SAME"
    h.Draw(drawopt)
    legend.AddEntry(h, label, "l")

legend.Draw()

c.Update()
c.SaveAs("../../idea_fullsim/fast_sim/histograms_view/p8_ee_WW_ecm365_fullhad/histograms_jetE_sorted_overlay.pdf")
print("Done")

