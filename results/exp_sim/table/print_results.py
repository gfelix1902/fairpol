import utils
import pandas as pd
import numpy as np
import joblib

if __name__ == "__main__":
    scale = 10
    round_digits = 2
    m = "dr"
    path = utils.get_project_path() + "/results/exp_sim/table/"
    pvalues = joblib.load(path + "pvalues_" + m + ".pkl")
    pvalues0 = joblib.load(path + "pvalues0_" + m + ".pkl")
    pvalues1 = joblib.load(path + "pvalues1_" + m + ".pkl")
    pvalues_diff_val = np.abs(pvalues1.values - pvalues0.values)
    pvalues_diff = pd.DataFrame(data=pvalues_diff_val, columns=pvalues.columns)
    af = joblib.load(path + "af_" + m + ".pkl")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Policy values Mean")
        print(round(pvalues.mean() * scale, round_digits))
        print("Policy values 1 Mean")
        print(round(pvalues1.mean() * scale, round_digits))
        print("Policy values 0 Mean")
        print(round(pvalues0.mean() * scale, round_digits))
        print("Action fairness Mean")
        print(round(af.mean() * scale, round_digits))
        print("Difference Mean")
        print(round(pvalues_diff.mean() * scale, round_digits))

        print("Policy values SD")
        print(round(pvalues.std() * scale, round_digits))
        print("Policy values 1 SD")
        print(round(pvalues1.std() * scale, round_digits))
        print("Policy values 0 SD")
        print(round(pvalues0.std() * scale, round_digits))
        print("Action fairness SD")
        print(round(af.std() * scale, round_digits))
        print("Difference SD")
        print(round(pvalues_diff.std() * scale, round_digits))