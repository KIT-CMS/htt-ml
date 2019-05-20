import ROOT as r
import glob
import yaml
import sys

r.gROOT.SetBatch()
r.gStyle.SetPalette(r.kFuchsia) #kBlueGreenYellow kCoffee
r.gStyle.SetPaintTextFormat(".2f")
r.TColor.InvertPalette()

if len(sys.argv) != 2:
    print 'Please provide exactly one argument being a wildcard path to the .yaml inputs, e.g.: python plot_root_confusion.py "/ekpwww/web/wunsch/public_html/2019-02-11/confusion_matrices/*/*.yaml"'
    exit(1)

wildcard = sys.argv[1]

input_list = [f for f in glob.glob(wildcard) if "efficiency" in f and "fold0" in f]

channel_dict = {
    "em": "e#mu",
    "et": "e_{}#tau_{h}",
    "mt": "#mu_{}#tau_{h}",
    "tt": "#tau_{h}#tau_{h}"
}

classes_true = {
    "em" : [{"ggH":"em_ggh"},{"qqH":"em_qqh"},{"ztt":"em_ztt"},{"qcd":"em_ss"},   {"tt":"em_tt"},{"misc":"em_misc"}, {"db":"em_vv"}, {"st":"em_st"}],
    "et" : [{"ggH":"ggh"},{"qqH":"qqh"},{"ztt":"ztt"},{"qcd":"ss"},   {"tt":"tt"},{"misc":"misc"},{"zll":"zll"},{"wj":"w"} ],
    "mt" : [{"ggH":"ggh"},{"qqH":"qqh"},{"ztt":"ztt"},{"qcd":"ss"},   {"tt":"tt"},{"misc":"misc"},{"zll":"zll"},{"wj":"w"} ],
    "tt" : [{"ggH":"ggh"},{"qqH":"qqh"},{"ztt":"ztt"},{"qcd":"noniso"},            {"misc":"misc"}],
}

classes_pred = {}

for ch in classes_true:
    classes_pred[ch] = list(reversed(classes_true[ch]))


for inp in input_list:
    year, ch = inp.split("/")[7].split("_")
    with open(inp,"r") as f:
        inp_dict = yaml.load(f.read())
    f.close()

    canv = r.TCanvas("confusion", "NN confusion matrix", 600, 600)
    canv.SetGridx(1)
    canv.SetLogx(0)
    canv.SetGridy(1)
    canv.SetLogy(0)
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.05)
    canv.SetTopMargin(0.05)
    canv.SetBottomMargin(0.15)

    nbins = len(classes_true[ch])

    hc = r.TH2F("hc","", nbins, 0.,nbins*1.,nbins, 0.,nbins*1.)
    hc.SetXTitle("True event class")
    hc.GetXaxis().CenterTitle()
    hc.GetXaxis().SetLabelFont(132) # 42
    hc.GetXaxis().SetLabelSize(0.055)
    hc.GetXaxis().SetLabelOffset(0.005)
    hc.GetXaxis().SetTitleSize(0.045)
    hc.GetXaxis().SetTitleFont(42)
    hc.GetXaxis().SetTitleColor(1)
    hc.GetXaxis().SetTitleOffset(1.5)

    hc.SetYTitle("NN predicted event class")
    hc.GetYaxis().CenterTitle()
    hc.GetYaxis().SetLabelFont(132) # 42
    hc.GetYaxis().SetLabelSize(0.055)
    hc.GetYaxis().SetLabelOffset(0.005)
    hc.GetYaxis().SetTitleSize(0.045)
    hc.GetYaxis().SetTitleFont(42)
    hc.GetYaxis().SetTitleColor(1)
    hc.GetYaxis().SetTitleOffset(1.5)

    for i in range(nbins):
        hc.GetXaxis().SetBinLabel(i+1, classes_true[ch][i].keys()[0])
        hc.GetYaxis().SetBinLabel(i+1, classes_pred[ch][i].keys()[0])

    hc.SetMaximum(2.)
    hc.LabelsOption("v")
    hc.SetStats(0)

    for i in range(nbins):
        for j in range(nbins):
            value = inp_dict[classes_true[ch][i].values()[0]][classes_pred[ch][j].values()[0]]
            hc.Fill(classes_true[ch][i].keys()[0], classes_pred[ch][j].keys()[0], value)

    hc.SetMarkerSize(1.5)
    hc.Draw("textcol")
    canv.RedrawAxis()

    tex = r.TLatex()
    tex.SetNDC()
    tex.SetLineWidth(2)
    tex.SetTextAlign(11)
    tex.SetTextSize(20)
    tex.SetTextFont(43)
    tex.DrawLatex(0.17, 0.965, "%s (%s)"%(channel_dict[ch], year))
    tex.SetTextSize(25)
    tex.SetTextFont(43)
    tex.DrawLatex(0.46, 0.965, "CMS")
    tex.SetTextFont(53)
    tex.DrawLatex(0.56, 0.965, "Simulation Preliminary")

    canv.Update()
    canv.Print("%s_%s_confusion.pdf"%(ch,year))
    canv.Print("%s_%s_confusion.png"%(ch,year))
