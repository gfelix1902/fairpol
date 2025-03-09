import numpy as np
import pandas as pd
import utils
from data.data_structures import Static_Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_oregon(config, standardize = True):
    # Select configuration file here
    path = utils.get_project_path() + "/data/oregon_health_exp/OHIE_Data/"
    data_descr = pd.read_stata(path + "oregonhie_descriptive_vars.dta")
    data_state = pd.read_stata(path + "oregonhie_stateprograms_vars.dta")[["person_id", "ohp_all_ever_matchn_30sep2009"]]
    data_ed = pd.read_stata(path + "oregonhie_ed_vars.dta")[["person_id", "num_visit_pre_cens_ed"]]
    #outcome = pd.read_stata(path + "oregonhie_inperson_vars.dta")[["person_id", "health_change_inp"]]
    outcome = pd.read_stata(path + "oregonhie_survey12m_vars.dta")[["person_id", "cost_any_owe_12m", "cost_tot_owe_12m"]]
    #Preprocess desciptive data
    #Number of signed up people
    data_descr.loc[data_descr.numhh_list == "signed self up", "num_signed_up"] = 1
    data_descr.loc[data_descr.numhh_list == "signed self up + 1 additional person", "num_signed_up"] = 2
    data_descr.loc[data_descr.numhh_list == "signed self up + 2 additional people", "num_signed_up"] = 3
    #Week of sign up
    data_descr.loc[data_descr.week_list == "Week 1", "week_signup"] = 1
    data_descr.loc[data_descr.week_list == "Week 2", "week_signup"] = 2
    data_descr.loc[data_descr.week_list == "Week 3", "week_signup"] = 3
    data_descr.loc[data_descr.week_list == "Week 4", "week_signup"] = 4
    data_descr.loc[data_descr.week_list == "Week 5", "week_signup"] = 5
    #Age
    data_descr["age"] = 2009 - data_descr["birthyear_list"]
    #Gender
    data_descr["gender"] = 1
    data_descr["gender"][data_descr.female_list == "0: Male"] = 0
    #City
    data_descr["city"] = 0
    data_descr["city"][data_descr.zip_msa_list == "Zip code of residence in a MSA"] = 1
    #Language
    data_descr["language"] = 0
    data_descr["language"][data_descr.english_list == "Requested English materials"] = 1
    data_descr = data_descr.merge(data_ed, on="person_id")
    data_descr = data_descr.merge(data_state, on="person_id")
    data_descr = data_descr.merge(outcome, on="person_id")
    #Treatment
    data_descr["treat"] = 0
    #data_descr["treat"][data_descr.ohp_all_ever_matchn_30sep2009 == "Enrolled"] = 1
    data_descr["treat"][data_descr.ohp_all_ever_matchn_30sep2009 == "Enrolled"] = 1
    #Health outcome
    outcome_type = "continuous"
    data_descr["outcome"] = data_descr["cost_tot_owe_12m"]
    data_descr["outcome"][data_descr.cost_any_owe_12m == "No"] = 0
    cond1 = data_descr["cost_tot_owe_12m"].isnull()
    cond2 = data_descr.cost_any_owe_12m == "Yes"
    data_descr["outcome"][cond1 & cond2] = data_descr["cost_tot_owe_12m"].median()

    #Only use necissary covariates
    data_labels = ["outcome", "treat", "age", "num_signed_up", "week_signup", "num_visit_pre_cens_ed", "gender", "language", "city"]
    data = data_descr[data_labels]
    #Remove NA values
    imp = IterativeImputer(max_iter=10, random_state=0)
    data = pd.DataFrame(imp.fit_transform(X=data), columns=data_labels)
    #data = data.dropna(axis=0)
    #Transform with sigmoid
    data["outcome"] = -np.log(1 + data.outcome)
    #Train/ Val/ Test split
    f_train = config["train_frac"]
    f_val = config["val_frac"]
    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), [int(f_train * len(data)), int((f_train + f_val) * len(data))])
    #Outcome + treatment
    y_train = np.expand_dims(df_train["outcome"].values, axis=1)
    a_train = np.expand_dims(df_train["treat"].values, axis=1)
    y_val = np.expand_dims(df_val["outcome"].values, axis=1)
    a_val = np.expand_dims(df_val["treat"].values, axis=1)
    y_test = np.expand_dims(df_test["outcome"].values, axis=1)
    a_test = np.expand_dims(df_test["treat"].values, axis=1)
    #Sensitive attribute to one hot encoding
    s_train = np.expand_dims(df_train["gender"].values, axis=1)
    s_val = np.expand_dims(df_val["gender"].values, axis=1)
    s_test = np.expand_dims(df_test["gender"].values, axis=1)
    enc_s = OneHotEncoder().fit(s_train)
    s_train = enc_s.transform(s_train).toarray()
    s_val = enc_s.transform(s_val).toarray()
    s_test = enc_s.transform(s_test).toarray()

    x_names = ["age", "num_signed_up", "week_signup", "num_visit_pre_cens_ed", "language"]
    x_types = ["continuous", "continuous", "continuous", "continuous", "binary"]
    #Unsensitive covariates
    x_train = df_train[x_names].values
    x_val = df_val[x_names].values
    x_test = df_test[x_names].values
    #Create datasets
    d_train = Static_Dataset(y=y_train, a=a_train, x=x_train, s=s_train, y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_val = Static_Dataset(y=y_val, a=a_val, x=x_val, s=s_val, y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_test = Static_Dataset(y=y_test, a=a_test, x=x_test, s=s_test, y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    if standardize:
        d_train.standardize()
        d_val.standardize()
        d_test.standardize()
        #d_train.data["y"] = - d_train.data["y"]
        #d_val.data["y"] = - d_val.data["y"]
        #d_test.data["y"] = - d_test.data["y"]
    d_train.convert_to_tensor()
    d_val.convert_to_tensor()
    d_test.convert_to_tensor()

    return {"d_train": d_train, "d_val": d_val, "d_test": d_test}




if __name__ == "__main__":
    load_oregon()


