"""Build an English, compilable results report from verified tables."""
from pathlib import Path
import pandas as pd
from summarize_campaign import REPORT,LABELS

def main():
    df=pd.read_csv(REPORT/'overall_metrics.csv').query("weighting=='design'").set_index('model')
    comparisons=pd.read_csv(REPORT/'paired_family_bootstrap.csv')
    c=comparisons.query("model=='blend' and baseline=='host'").iloc[0]
    f=comparisons.query("model=='fusion' and baseline=='host'").iloc[0]
    z=comparisons.query("model=='blend' and baseline=='zero'").iloc[0]
    t=comparisons.query("model=='blend' and baseline=='tree_dual_l2'").iloc[0]
    l1=comparisons.query("model=='tree_dual_l1' and baseline=='host'").iloc[0]
    mae=100*(1-df.loc['blend','mae']/df.loc['host','mae'])
    results=[r'\section{Forecasting Results and Interpretation}',
        'The complete campaign comprises 70 neural fits and 75 tree fits. Each of the 1,633 county--event pairs '
        'is evaluated exactly once in an outer event fold. All results below use the fixed public development panel; '
        'the sealed confirmation outcomes were not accessed.',
        r'\begin{table}[ht]\centering\small',
        r'\caption{Full 144-hour trajectory accuracy: design-weighted pooled metrics.}',
        r'\begin{tabular}{lrr}\toprule Model & MAE & RMSE \\\midrule']
    for n,label in LABELS.items():results.append(f"{label} & {df.loc[n,'mae']:.5f} & {df.loc[n,'rmse']:.5f}"+r' \\')
    results += [r'\bottomrule\end{tabular}\end{table}',
        f'The inner-selected ensemble changes pooled RMSE relative to the matched host by '
        f'{-c.rmse_gain_pct:+.2f}\\% (negative is better), with a paired family-bootstrap 95\\% interval '
        f'[{ -c.upper95:+.2f}\\%, { -c.lower95:+.2f}\\%]. It improves RMSE on {int(c.systems_improved)} of 15 systems. '
        f'The corresponding MAE change is {-mae:+.2f}\\%.',
        f'The joint GCRK-rate predictor changes RMSE relative to the matched host by '
        f'{-f.rmse_gain_pct:+.2f}\\%, with a 95\\% interval [{-f.upper95:+.2f}\\%, {-f.lower95:+.2f}\\%]. '
        'Because this arm changes the kernel and rate heads together, this is not an isolated test of geographic information.']
    if c.lower95>0:
        results.append('The interval supports a positive ensemble gain within this development evaluation. '
                       'It does not remove uncertainty from prior development choices or establish independent confirmation.')
    else:
        results.append('The ensemble comparison does not establish a reliably positive gain at the stated interval level. '
                       'Any favorable point estimate must therefore be interpreted as exploratory.')
    results += [f'Against zero prediction, the ensemble reduces pooled RMSE by {z.rmse_gain_pct:.2f}\\% '
        f'(95\\% interval [{z.lower95:.2f}\\%, {z.upper95:.2f}\\%]), but improves only '
        f'{int(z.systems_improved)} of 15 systems. Its RMSE reduction over the dual-weather residual tree is '
        f'{t.rmse_gain_pct:.2f}\\% (interval [{t.lower95:.2f}\\%, {t.upper95:.2f}\\%]).',
        f'The lowest full-path MAE belongs to the dual-weather absolute-loss tree '
        f'({df.loc["tree_dual_l1","mae"]:.5f}); its MAE reduction relative to the host is '
        f'{l1.mae_gain_pct:.2f}\\% (interval [{l1.mae_lower95:.2f}\\%, {l1.mae_upper95:.2f}\\%]). '
        'Thus the RMSE-oriented ensemble and the absolute-loss tree represent different error tradeoffs. '
        'The present joint GCRK-rate arm does not support replacing the matched host.']
    figures=[r'\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/forecast_metrics.pdf}',
        r'\caption{Full-path errors for all fixed candidates. The ensemble weights were selected only from inner event-fold predictions.}',
        r'\end{figure}',
        r'\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/storm_transfer.pdf}',
        r'\caption{Every tropical development system, including weak systems. Both horizontal axes use logarithmic scales.}',
        r'\end{figure}']
    (REPORT/'PAPER_RESULTS.tex').write_text('\n\n'.join(results)+'\n')
    (REPORT/'report_horizon_table.tex').write_text((REPORT/'results_table.tex').read_text()
        .replace(r'\begin{table*}[t]',r'\begin{table}[!ht]')
        .replace(r'\end{table*}',r'\end{table}'))
    main=[r'\documentclass[10pt]{article}',r'\usepackage[margin=0.85in]{geometry}',
        r'\usepackage{amsmath,booktabs,multirow,graphicx,hyperref}',r'\setlength{\tabcolsep}{4pt}',
        r'\title{Weather-Conditioned Tropical Outage Prediction\\\large Development Experiment Report}',
        r'\author{}',r'\date{September 27, 2026}',r'\begin{document}\maketitle',
        r'\noindent\textbf{Status.} Completed development evaluation; not an independent confirmation or an operational weather-forecast test.',
        r'\input{PAPER_RESULTS.tex}',r'\clearpage',r'\input{report_horizon_table.tex}',
        'The four horizons are point predictions from the same 144-hour path; weights do not vary by horizon. '
        'The full-path RMSE ordering need not hold at every lead time. Persistence is particularly strong '
        'at the earliest leads in this panel. All seed, per-system and weighting-sensitivity results accompany the report.',
        r'\section{Methods}',r'\input{../PAPER_METHODS.tex}',r'\clearpage',*figures,r'\end{document}']
    (REPORT/'development_report.tex').write_text('\n'.join(main)+'\n')
    print('Generated result text and compilable report source')

if __name__=='__main__':main()
