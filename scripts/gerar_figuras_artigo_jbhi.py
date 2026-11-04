import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = sys.argv[1]; OUT = sys.argv[2]
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["Times New Roman","Times","Nimbus Roman"],"mathtext.fontset":"stix","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,
                     "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"figure.dpi":300})
C1, C2, C0 = "#1f5f9e", "#4fb3bf", "#9a9a9a"

# ---------- Fig 1: variance structure ----------
sd_p = []
for ch in pd.read_csv(f"{D}/PulseDB_features/_features_pulsedb_calbased_test_v3.csv",
                      usecols=["subject","record","bp_sys_mean"], chunksize=500_000):
    sd_p.append(ch.groupby(["subject","record"], as_index=False)["bp_sys_mean"].first())
seg = pd.concat(sd_p).groupby(["subject","record"], as_index=False)["bp_sys_mean"].first()
sdp = seg.groupby("subject")["bp_sys_mean"].std(ddof=1).dropna()
ptt = pd.read_csv(f"{D}/PTT_PPG/_features_v3_clean.csv", usecols=["subject","condition","bp_sys_mean"])
pseg = ptt.groupby(["subject","condition"], as_index=False)["bp_sys_mean"].first()
sdt = pseg.groupby("subject")["bp_sys_mean"].std(ddof=1).dropna()
va = pd.read_csv(f"{D}/PulseDB_features/_nb13_auditoria_variancia.csv")
icc = {r.base: r.ICC for r in va[va.alvo=="PAS"].itertuples()}

fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.6))
ax[0].hist(sdp, bins=45, density=True, color=C1, alpha=.85, label=f"PulseDB/VitalDB (n = {len(sdp)})")
ax[0].hist(sdt, bins=12, density=True, color="#d95f02", alpha=.75, label=f"PTT-PPG (n = {len(sdt)})")
ax[0].axvline(sdp.median(), color=C1, ls="--", lw=1); ax[0].axvline(sdt.median(), color="#d95f02", ls="--", lw=1)
ax[0].set_xlabel("Within-subject SD of reference SBP (mmHg)"); ax[0].set_ylabel("Density")
ax[0].legend(frameon=False); ax[0].set_title("(a) Within-subject SBP variability")
names = ["PTT-PPG", "PulseDB/VitalDB"]
between = [icc["PTT-PPG"]*100, icc["PulseDB (VitalDB)"]*100]
ax[1].bar(names, between, color="#7fcdbb", label="Between subjects")
ax[1].bar(names, [100-b for b in between], bottom=between, color="#253494", label="Within subject")
for i,b in enumerate(between): ax[1].text(i, b/2, f"ICC = {b/100:.2f}", ha="center", va="center", fontsize=7)
ax[1].set_ylabel("Share of total SBP variance (%)"); ax[1].set_ylim(0,100); ax[1].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5,-0.42))
ax[1].set_title("(b) Variance decomposition")
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_variance.pdf"); plt.close(fig)

# ---------- Fig 2: main result + permutation ----------
rp = pd.read_csv(f"{D}/PulseDB_features/_nb13_resultado_principal.csv")
nul = pd.read_csv(f"{D}/PulseDB_features/_nb13_distribuicao_nula.csv")
perm = pd.read_csv(f"{D}/PulseDB_features/_nb13_permutacao.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw={"width_ratios":[1.1,1.1,1.3]})
lab = ["Calibration\nmean", "Ridge", "Random\nForest"]
for k, alvo in enumerate(["PAS","PAD"]):
    r = rp[rp.alvo==alvo]
    y = r.MAE.values; err = np.vstack([y-r.MAE_lo.values, r.MAE_hi.values-y])
    ax[k].bar(lab, y, yerr=err, capsize=3, color=[C0,C1,C2], width=.65)
    for i,(m,r2) in enumerate(zip(y, r.R2.values)):
        ax[k].text(i, m+0.35, f"{m:.2f}\n$R^2$ {r2:+.3f}", ha="center", va="bottom", fontsize=6.3)
    ax[k].set_ylim(0, y.max()*1.38); ax[k].set_ylabel("MAE of $\\Delta$%s (mmHg)" % ("SBP" if alvo=="PAS" else "DBP"))
    ax[k].set_title("(%s) $\\Delta$%s, held-out block" % ("ab"[k], "SBP" if alvo=="PAS" else "DBP"))
ax[2].hist(nul.PAS, bins=40, color=C0, label="Null (1000 within-subject\nlabel permutations)")
ax[2].axvline(perm[perm.alvo=="PAS"].R2_real.iloc[0], color="#c0392b", lw=1.6, label="Observed $R^2$ = %.3f" % perm[perm.alvo=="PAS"].R2_real.iloc[0])
ax[2].set_xlabel("$R^2$ for $\\Delta$SBP"); ax[2].set_ylabel("Count"); ax[2].legend(frameon=False, fontsize=6.2, loc="upper center")
ax[2].set_title("(c) Permutation control (z = +%.0f)" % perm[perm.alvo=="PAS"].z.iloc[0])
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.pdf"); plt.close(fig)

# ---------- Fig 3: ablations ----------
ab = pd.read_csv(f"{D}/PulseDB_features/_nb13_ablacoes.csv")
names = {"Todas as 41 features":"All 41 features","SEM redundancias exatas":"Without exact duplicates (39)",
 "SEM escala — Versao C":"Scale-invariant only (35)","SEM frequencia cardiaca":"Without heart rate (39)",
 "REDUZIDO SEM DUPLICATAS":"Reduced set (13)","SO as 6 de escala":"Only the 6 scale-dependent",
 "SO estatisticas/Hjorth":"Only statistics + Hjorth (8)","SO amplitude":"Only amplitude (4)",
 "SO larguras de pulso":"Only pulse widths (24)","SO tempos sistole/diastole":"Only systolic/diastolic times (3)",
 "SO frequencia cardiaca":"Only heart rate (2)"}
ab = ab[ab.conjunto.isin(names)].copy(); ab["n"] = ab.conjunto.map(names)
ab = ab.sort_values("R2_PAS")
fig, ax = plt.subplots(figsize=(3.5, 2.6))
yy = np.arange(len(ab)); h=.38
ax.barh(yy+h/2, ab.R2_PAS, h, color=C1, label="$\\Delta$SBP"); ax.barh(yy-h/2, ab.R2_PAD, h, color=C2, label="$\\Delta$DBP")
ax.set_yticks(yy); ax.set_yticklabels(ab.n, fontsize=6.3); ax.set_xlabel("Held-out $R^2$ (deployment protocol)")
ax.legend(frameon=False, loc="lower right"); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.pdf"); plt.close(fig)

# ---------- Fig 4: coefficients (top 12 by |coef| for SBP) ----------
co = pd.read_csv(f"{D}/PulseDB_features/_nb13_coeficientes_reduzido.csv")
pretty = {"std":"SD of cycle","hjorth_mobility":"Hjorth mobility","systolic_upstroke_time_s":"Systolic upstroke time",
 "cardiac_period_s":"Cardiac period","hjorth_complexity":"Hjorth complexity","amplitude":"Peak-to-foot amplitude",
 "skewness":"Skewness","variance":"Variance","heart_rate_bpm":"Heart rate","valley_value":"Foot value",
 "auc":"Area under cycle","sut_dt_ratio":"Upstroke/diastolic ratio","crest_factor":"Crest factor","w50_ratio":"Width ratio 50%",
 "w75_ratio":"Width ratio 75%","w66_dia":"Diastolic width 66%","w66_sys":"Systolic width 66%","w75_dia":"Diastolic width 75%"}
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=False)
for k, alvo in enumerate(["PAS","PAD"]):
    c = co[co.alvo==alvo].sort_values("abs_medio", ascending=False).head(12).sort_values("coef_medio")
    cols = [C1 if v>0 else "#c0392b" for v in c.coef_medio]
    ax[k].barh([pretty.get(f,f) for f in c.feature], c.coef_medio, xerr=c.coef_sd, color=cols, capsize=2, error_kw={"lw":.7})
    ax[k].axvline(0, color="k", lw=.6); ax[k].set_xlabel("Ridge coefficient, standardized features (mmHg)")
    ax[k].set_title("(%s) $\\Delta$%s" % ("ab"[k], "SBP" if alvo=="PAS" else "DBP")); ax[k].tick_params(axis="y", labelsize=6.3)
    ax[k].spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_coef.pdf"); plt.close(fig)
print("figs ok", len(sdp), len(sdt), sdp.median(), sdt.median())
