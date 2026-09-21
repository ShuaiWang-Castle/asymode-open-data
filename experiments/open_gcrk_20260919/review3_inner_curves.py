"""Inner learning curves of W and GCRK at matched steps (round 1, county design, 25 cell pairs): training
loss and held-out inner loss of GCRK relative to W. Writes results/review3_inner_curves.csv."""
import numpy as np, pandas as pd
import common as C
rows = []
for s in C.SEEDS:
    for f in C.load_splits()["main"]:
        w = pd.read_csv(C.cell("main", s, f, "W") / "selection_trace.csv").set_index("step")
        g = pd.read_csv(C.cell("main", s, f, "GCRK") / "selection_trace.csv").set_index("step")
        fit = [c for c in w.columns if c.endswith("_fit_loss")]
        for st in (100, 200, 300, 400):
            if st in w.index and st in g.index:
                rows.append(dict(seed=s, fold=f, step=st, inner_rel=g.pooled_inner_mse[st] / w.pooled_inner_mse[st] - 1,
                                 train_rel=g.loc[st, fit].mean() / w.loc[st, fit].mean() - 1))
d = pd.DataFrame(rows); d.to_csv(C.RESULTS / "review3_inner_curves.csv", index=False)
print(d.groupby("step").agg(inner_rel=("inner_rel", "mean"), inner_lower=("inner_rel", lambda x: int((x < 0).sum())),
                            train_rel=("train_rel", "mean"), train_lower=("train_rel", lambda x: int((x < 0).sum())), n=("step", "size")).round(4))
