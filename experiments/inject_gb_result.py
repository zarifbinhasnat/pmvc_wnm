"""
inject_gb_result.py — appends the pre-measured gradient_boosting_capped /
View A result (from the killed exp6b run's log, verified real output, not
re-run to save ~22 minutes of redundant compute) into the exp6 results CSV.

Source values (verbatim from experiments/../exp6b.output):
  boosting  gradient_boosting_capped | CV f1=0.6006+/-0.0064 | holdout f1=0.5251+/-0.0061
  | brier=0.1932 | fit=7.12s | wall=1324.4s
"""
import pandas as pd

df = pd.read_csv("results/exp6_exhaustive_model_zoo.csv")

row = dict(view="A", family="boosting", model="gradient_boosting_capped",
          cv_f1_mean=0.6006, cv_f1_std=0.0064, cv_fit_s=7.12,
          f1_mean=0.5251, f1_std=0.0061, acc_mean=None,
          fit_s_mean=7.12, brier_mean=0.1932, total_wall_s=1324.4)

df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_csv("results/exp6_exhaustive_model_zoo.csv", index=False)
print(f"Injected gradient_boosting_capped/View A. Total rows now: {len(df)}")
print(df[df.model == "gradient_boosting_capped"].to_string(index=False))
