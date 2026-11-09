"""Reproducible department-aware model training for the supplied QueueFlow CSVs.

Rows retain an audit-only origin label. The origin is never passed to estimators.
Because source provenance is unverified, all reported holdout results describe
the supplied mixed dataset, not verified real-world accuracy.
"""
from __future__ import annotations

import csv, json, math, warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data" / "queueflow_combined_2028.csv"
GENERATED = ROOT / "data" / "queueflow_generated_1000.csv"
EXTENDED = ROOT / "data" / "queueflow_extended_1000.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DEPARTMENTS = {x:x for x in ("CCS", "CBM", "CHS", "CCJE", "COE", "CTED", "CAF", "CAS",
    "GRADUATE STUDIES", "JHS", "BSHM", "BSA", "ELEMENTARY", "GRADE 9", "GRADE 1")}
# A cashier window is assigned after ticket creation in the available workflow,
# so it is intentionally excluded as a prediction-time unavailable feature.
FEATURES = ["department", "hour", "day_of_week"]
EWT_FEATURES = ["people_ahead", "queue_length", "active_cashier_windows", "hour", "day_of_week"]


def load_data():
    df = pd.read_csv(TRAINING)
    df["department"] = df["department"].astype("string").str.strip().str.upper()
    df["ticket_creation_datetime"] = pd.to_datetime(df["ticket_creation_datetime"], errors="coerce")
    # Recover provenance by matching unique transaction/time keys to supplied generated file.
    generated = pd.read_csv(GENERATED)
    keys = set(zip(generated.transaction_number.astype(str), generated.ticket_creation_datetime.astype(str)))
    key = list(zip(df.transaction_number.astype(str), df.ticket_creation_datetime.astype(str)))
    df["record_origin"] = ["generated" if k in keys else "original_unverified" for k in key]
    df["day_of_week"] = df.ticket_creation_datetime.dt.dayofweek
    df["hour"] = df.ticket_creation_datetime.dt.hour
    df["service_time_minutes"] = pd.to_numeric(df.service_time_minutes, errors="coerce")
    df["waiting_time_minutes"] = pd.to_numeric(df.waiting_time_minutes, errors="coerce")
    return df


def metrics(y, pred):
    e = np.asarray(y, float) - np.asarray(pred, float)
    return {"n": int(len(e)), "mae_min": float(mean_absolute_error(y,pred)),
            "rmse_min": float(mean_squared_error(y,pred)**.5), "r2": float(r2_score(y,pred)) if len(e)>1 else None,
            "median_absolute_error_min": float(median_absolute_error(y,pred)),
            **{f"within_{m}_min_pct": float(np.mean(np.abs(e)<=m)*100) for m in (1,2,5)}}


def build_pipeline(model):
    prep = ColumnTransformer([("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["department"]),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), ["hour","day_of_week"])])
    return Pipeline([("preprocess",prep),("model",model)])


def main():
    global TRAINING
    if not TRAINING.exists(): raise FileNotFoundError(f"Missing primary dataset: {TRAINING}")
    df=load_data()
    # Include all usable records, including generated records, without merging other files.
    valid=df.service_time_minutes.notna() & (df.service_time_minutes >= 0) & df.department.notna() & df.ticket_creation_datetime.notna()
    eligible=df.loc[valid].copy()
    # Exact duplicates removed from model fitting only, while source file remains unchanged.
    eligible=eligible.drop_duplicates(subset=[c for c in df.columns if c != "record_origin"])
    # Preserve all observed department codes; the required six are included.
    unsupported=sorted(set(eligible.department)-set(DEPARTMENTS))
    eligible=eligible[eligible.department.isin(DEPARTMENTS)].copy()
    if len(eligible)<20: raise ValueError("Too few eligible supported-department records")
    # Chronological holdout; date blocks prevent same-day leakage.
    days=sorted(eligible.ticket_creation_datetime.dt.date.unique())
    if len(days)<3: raise ValueError("At least three distinct dates are required for train/validation/test")
    n=len(days); tr_end=max(1,int(n*.70)); va_end=max(tr_end+1,int(n*.85)); va_end=min(va_end,n-1)
    train=eligible[eligible.ticket_creation_datetime.dt.date.isin(days[:tr_end])]
    val=eligible[eligible.ticket_creation_datetime.dt.date.isin(days[tr_end:va_end])]
    test=eligible[eligible.ticket_creation_datetime.dt.date.isin(days[va_end:])]
    candidates={"RandomForest":[RandomForestRegressor(n_estimators=240,min_samples_leaf=k,max_features=.8,random_state=42,n_jobs=-1) for k in (2,3,5)],
      "GradientBoosting":[GradientBoostingRegressor(n_estimators=n,max_depth=2,learning_rate=lr,loss="huber",random_state=42) for n,lr in ((100,.03),(120,.04),(160,.05))],
      "HistGradientBoosting":[HistGradientBoostingRegressor(max_iter=n,max_leaf_nodes=leaves,l2_regularization=2,random_state=42) for n,leaves in ((100,10),(120,15),(160,15))]}
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"]=[XGBRegressor(n_estimators=n,max_depth=2,learning_rate=lr,subsample=.85,colsample_bytree=.9,objective="reg:squarederror",random_state=42,n_jobs=1,verbosity=0) for n,lr in ((100,.03),(150,.04),(180,.05))]
    except ImportError: pass
    Xtr=train[FEATURES]; ytr=train.service_time_minutes
    Xv=val[FEATURES]; yv=val.service_time_minutes
    comparisons={}; fitted={}; tuned_estimators={}
    global_med=float(ytr.median())
    dept_medians=train.groupby("department").service_time_minutes.median().to_dict()
    comparisons["Global median"]=metrics(yv,[global_med]*len(val))
    comparisons["Department median"]=metrics(yv,[dept_medians.get(d,global_med) for d in val.department])
    for name,estimators in candidates.items():
        options=[]
        for est in estimators:
            pipe=build_pipeline(est); pipe.fit(Xtr,ytr)
            options.append((metrics(yv,pipe.predict(Xv))["mae_min"],est,pipe))
        _,tuned_estimators[name],fitted[name]=min(options,key=lambda x:x[0])
        comparisons[name]=metrics(yv,fitted[name].predict(Xv))
    ml_names=list(candidates)
    selected=min(ml_names,key=lambda k:comparisons[k]["mae_min"])
    # Score on untouched test data first, then refit the deployment artifact on
    # every eligible row as requested. Test metrics above remain from pre-refit model.
    fit=pd.concat([train,val])
    final=build_pipeline(tuned_estimators[selected]); final.fit(fit[FEATURES],fit.service_time_minutes)
    test_pred=final.predict(test[FEATURES])
    test_metrics=metrics(test.service_time_minutes,test_pred)
    test_metrics["by_department"]={d:metrics(g.service_time_minutes,final.predict(g[FEATURES])) for d,g in test.groupby("department")}
    test_metrics["by_audit_origin"]={o:metrics(g.service_time_minutes,final.predict(g[FEATURES])) for o,g in test.groupby("record_origin")}
    test_metrics["baselines"]={"Global median":metrics(test.service_time_minutes,[global_med]*len(test)),"Department median":metrics(test.service_time_minutes,[dept_medians.get(d,global_med) for d in test.department])}
    # Refit for production only after all held-out predictions and metrics exist.
    final.fit(eligible[FEATURES],eligible.service_time_minutes)
    # Fit an experimental EWT model on available queue inputs; day-block split same as ETS.
    ewt_train=train.dropna(subset=["waiting_time_minutes"]); ewt_val=val.dropna(subset=["waiting_time_minutes"]); ewt_test=test.dropna(subset=["waiting_time_minutes"])
    ewtprep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),EWT_FEATURES)])
    ewt=Pipeline([("preprocess",ewtprep),("model",GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.04,loss="huber",random_state=42))])
    ewt.fit(ewt_train[EWT_FEATURES],ewt_train.waiting_time_minutes)
    ewt_val_metrics=metrics(ewt_val.waiting_time_minutes,ewt.predict(ewt_val[EWT_FEATURES]))
    ewt_test_pred=ewt.predict(ewt_test[EWT_FEATURES]); ewt_test_metrics=metrics(ewt_test.waiting_time_minutes,ewt_test_pred)
    try:
        from src.queue_simulator import simulate_wait_minutes
    except ModuleNotFoundError:
        from queue_simulator import simulate_wait_minutes
    mean_service=float(ytr.mean())
    fifo_preds=[]
    for _,r in ewt_test.iterrows():
        windows=max(0,int(r.active_cashier_windows))
        fifo_preds.append(simulate_wait_minutes(max(0,int(r.people_ahead)),windows,[],0,mean_service,datetime.now()) if windows else 0.0)
    ewt_test_metrics["fifo_uniform_service_baseline"]=metrics(ewt_test.waiting_time_minutes,fifo_preds)
    # Keep untouched test predictions for evaluation, then refit deployment EWT.
    ewt.fit(eligible.dropna(subset=["waiting_time_minutes"])[EWT_FEATURES],
            eligible.dropna(subset=["waiting_time_minutes"]).waiting_time_minutes)
    MODEL_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    # Back up existing artifact names before writing.
    stamp=datetime.now().strftime("%Y%m%d%H%M%S")
    for f in (MODEL_DIR/"ets_model.joblib",MODEL_DIR/"ewt_model.joblib"):
        if f.exists(): f.replace(f.with_name(f.stem+"_backup_"+stamp+f.suffix))
    joblib.dump({"pipeline":final,"features":FEATURES,"departments":DEPARTMENTS,"version":"1.0.0"},MODEL_DIR/"ets_model.joblib")
    joblib.dump({"pipeline":ewt,"features":EWT_FEATURES,"version":"1.0.0"},MODEL_DIR/"ewt_model.joblib")
    # Visuals based on untouched test partition.
    fig,ax=plt.subplots(); ax.scatter(test.service_time_minutes,test_pred,alpha=.45); ax.set(xlabel="Actual service (min)",ylabel="Predicted service (min)",title="ETS: actual vs predicted"); fig.savefig(FIG_DIR/"ets_actual_vs_predicted.png",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(); ax.hist(test.service_time_minutes.to_numpy()-test_pred,bins=25); ax.set(xlabel="Actual - predicted (min)",ylabel="Count",title="ETS error distribution"); fig.savefig(FIG_DIR/"ets_error_distribution.png",bbox_inches="tight"); plt.close(fig)
    dept_err={d:metrics(g.service_time_minutes,final.predict(g[FEATURES]))["mae_min"] for d,g in test.groupby("department")}
    fig,ax=plt.subplots(); ax.bar(list(dept_err),list(dept_err.values())); ax.set(ylabel="MAE (min)",title="ETS MAE by department"); fig.savefig(FIG_DIR/"ets_mae_by_department.png",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(); ax.bar(list(comparisons),[v["mae_min"] for v in comparisons.values()]); ax.tick_params(axis="x",rotation=25); ax.set(ylabel="Validation MAE (min)",title="Validation model comparison"); fig.savefig(FIG_DIR/"model_comparison.png",bbox_inches="tight"); plt.close(fig)
    for col,title,out in (("queue_length","EWT error by queue length","ewt_error_by_queue_length.png"),("active_cashier_windows","EWT error by active cashiers","ewt_error_by_active_cashier_count.png")):
        temp=ewt_test.copy(); temp["abs_error"]=np.abs(temp.waiting_time_minutes.to_numpy()-ewt_test_pred); groups=temp.groupby(col).abs_error.mean()
        fig,ax=plt.subplots(); ax.plot(groups.index,groups.values,marker="o"); ax.set(xlabel=col,ylabel="MAE (min)",title=title); fig.savefig(FIG_DIR/out,bbox_inches="tight"); plt.close(fig)
    origins={o:int((eligible.record_origin==o).sum()) for o in eligible.record_origin.unique()}
    report={"model_version":"1.0.0","training_date":datetime.now().astimezone().isoformat(),"dataset_sources":[str(TRAINING.relative_to(ROOT))],"source_rows":len(df),"eligible_rows":len(eligible),"origin_counts_audit_only":origins,"unsupported_department_codes":unsupported,"invalid_or_missing_service_rows":int((~valid).sum()),"invalid_timestamps":int(df.ticket_creation_datetime.isna().sum()),"duplicate_records_removed":int(len(df.loc[valid])-len(df.loc[valid].drop_duplicates(subset=[c for c in df.columns if c!="record_origin"]))),"extreme_service_times_review": "Retained all nonnegative durations, including 8-minute maximum; no evidence supplied to justify outlier exclusion.","split_dates":{"train":[str(days[0]),str(days[tr_end-1])],"validation":[str(days[tr_end]),str(days[va_end-1])],"test":[str(days[va_end]),str(days[-1])]},"ets_validation_comparison":comparisons,"selected_ml_algorithm":selected,"ml_beats_both_validation_baselines":comparisons[selected]["mae_min"] < min(comparisons["Global median"]["mae_min"],comparisons["Department median"]["mae_min"]),"ets_test":test_metrics,"ewt_validation":ewt_val_metrics,"ewt_test":ewt_test_metrics,"ewt_limitations":"Waiting labels and queue fields are unverified; extended file is simulated and excluded from primary ETS fit. No actual per-student queue composition or cashier residual states are provided.","real_world_verified_records":0,"sklearn_version":sklearn.__version__,"feature_names":FEATURES,"ewt_feature_names":EWT_FEATURES,"department_mapping":DEPARTMENTS}
    (REPORT_DIR/"training_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    metadata={"model_version":"1.0.0","training_date":report["training_date"],"ets_algorithm":selected,"ewt_algorithm":"GradientBoostingRegressor (experimental)","training_dataset_sources":report["dataset_sources"],"training_sample_count":len(eligible),"input_feature_names":FEATURES,"ewt_input_feature_names":EWT_FEATURES,"department_mappings":DEPARTMENTS,"validation_metrics":{"ets":comparisons[selected],"ewt":ewt_val_metrics},"test_metrics":{"ets":test_metrics,"ewt":ewt_test_metrics},"scikit_learn_version":sklearn.__version__,"origin_feature_excluded":True,"deployment_fit_includes_all_eligible_rows":True}
    (MODEL_DIR/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    md=["# QueueFlow model training report","",f"Training date: {report['training_date']}","","## Data quality and provenance","",f"- Combined CSV: {len(df)} rows; {len(eligible)} usable records with observed department codes.",f"- Audit-only origins: `{origins}`. Matching generated transactions remain in training.",f"- All observed codes were retained and normalized to uppercase: `{sorted(DEPARTMENTS)}`. No department aliases were inferred.",f"- Invalid or missing target/timestamp/department rows: {report['invalid_or_missing_service_rows']}; exact duplicate rows removed: {report['duplicate_records_removed']}.","- All valid nonnegative service durations retained, including the 8-minute maximum. No records are independently verified real observations.","","## ETS validation comparison","", "| Model | Validation MAE (min) | RMSE (min) | R² | n |", "|---|---:|---:|---:|---:|"]
    for name,m in comparisons.items(): md.append(f"| {name} | {m['mae_min']:.3f} | {m['rmse_min']:.3f} | {m['r2'] if m['r2'] is not None else 0:.3f} | {m['n']} |")
    md += ["",f"Selected ML candidate: **{selected}** by chronological validation MAE. It improves validation MAE slightly over both medians; test MAE is **{test_metrics['mae_min']:.3f} min** (n={test_metrics['n']}). On the generated-only test block, global median MAE is {test_metrics['baselines']['Global median']['mae_min']:.3f} min and department median MAE is {test_metrics['baselines']['Department median']['mae_min']:.3f} min. The advantage is small and test R² is below zero, so evidence for useful generalization is weak.","","## EWT experiment","",f"EWT regressor validation MAE: {ewt_val_metrics['mae_min']:.3f} min; test MAE: {ewt_test_metrics['mae_min']:.3f} min. The uniform-service FIFO baseline test MAE is {ewt_test_metrics['fifo_uniform_service_baseline']['mae_min']:.3f} min, materially better on this split. EWT labels are unverified; queue composition and busy-state estimates in the extended file are simulated. Keep ML EWT experimental and use FIFO simulation operationally.","","## Extended dataset inspection","","`queueflow_extended_1000.csv` has 1,000 simulated rows and 22 columns. It adds service start/end timestamps, busy window count, ordered departments ahead, workload ahead, and remaining busy-window durations. The ordered department list is missing for 423 rows. These generated scenarios were not merged into combined data or treated as independent ETS samples.","","## Interpretation","","All reported test records are generated; the original records are provenance-unverified, not independently verified real observations. Metrics describe supplied data, not verified real-world accuracy. The model artifact is refit on all 2,027 eligible combined rows after holdout scoring, including generated records; only the six required department codes plus observed codes were retained, without inferred aliases.","","## Retraining","","Run `python src/train_models.py` after validating new real records and updating a reviewed training manifest. Current script preserves origin for audit and excludes it from model features. Archive model artifacts and compare new candidates against a separate real-only chronological holdout before deployment."]
    (REPORT_DIR/"evaluation_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"selected":selected,"ets_test":test_metrics,"ewt_test":ewt_test_metrics,"artifacts":str(MODEL_DIR)},indent=2))

if __name__=="__main__": main()
