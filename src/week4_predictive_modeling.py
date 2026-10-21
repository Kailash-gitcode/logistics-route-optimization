"""Week 4 predictive delivery-time modeling and capacity-constrained triage demo.

Run from any working directory with: python src/week4_predictive_modeling.py
Uses the existing cleaned delivery dataset; it creates no replacement dataset.
Scikit-learn estimators are preferred when available. Compact NumPy fallbacks
keep this analysis runnable in the restricted project environment.
"""
from __future__ import annotations
from pathlib import Path
import json, math, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/cleaned_logistics_data.csv"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "visualizations"
TARGET = "time_taken_min"
SEED = 42
NUMERIC = ["approx_delivery_distance_km", "vehicle_condition", "multiple_deliveries", "order_hour", "order_day", "delivery_person_age"]
CATEGORICAL = ["road_traffic_density", "weather_conditions", "type_of_vehicle", "city", "festival", "day_of_week"]

sys.path.insert(0, str(ROOT / "src"))
from modeling import NumpyLinearRegression, NumpyRandomForestRegressor, regression_metrics

def load_data():
    data = pd.read_csv(DATA, low_memory=False)
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    return data.dropna(subset=["order_date", TARGET]).sort_values("order_date").reset_index(drop=True)

def split_by_date(data, fraction=.20):
    dates = np.sort(data.order_date.dt.normalize().unique())
    if len(dates) < 10: raise ValueError("Need at least 10 distinct dates for chronological holdout and CV")
    boundary = max(1, min(len(dates)-1, int(len(dates)*(1-fraction))))
    cutoff = pd.Timestamp(dates[boundary])
    return data[data.order_date.dt.normalize() < cutoff].copy(), data[data.order_date.dt.normalize() >= cutoff].copy(), cutoff

def encode_fit(train, other):
    """Fit numeric medians/category vocabulary on training subset only."""
    numeric = [c for c in NUMERIC if c in train]
    categorical = [c for c in CATEGORICAL if c in train]
    a = train[numeric].apply(pd.to_numeric, errors="coerce")
    b = other[numeric].apply(pd.to_numeric, errors="coerce")
    medians = a.median().fillna(0)
    a = a.fillna(medians); b = b.fillna(medians)
    ca = train[categorical].astype("string").fillna("Unknown").replace("", "Unknown")
    cb = other[categorical].astype("string").fillna("Unknown").replace("", "Unknown")
    da = pd.get_dummies(ca, dtype=float)
    db = pd.get_dummies(cb, dtype=float).reindex(columns=da.columns, fill_value=0.)
    xa = pd.concat([a.reset_index(drop=True), da.reset_index(drop=True)], axis=1).to_numpy(float)
    xb = pd.concat([b.reset_index(drop=True), db.reset_index(drop=True)], axis=1).to_numpy(float)
    return xa, xb, list(a.columns)+list(da.columns)

def make_models():
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        return {
            "Linear Regression": lambda: LinearRegression(),
            "Decision Tree (depth 8)": lambda: DecisionTreeRegressor(max_depth=8,min_samples_leaf=30,random_state=SEED),
            "Random Forest (12 trees)": lambda: RandomForestRegressor(n_estimators=12,max_depth=8,min_samples_leaf=30,max_features="sqrt",random_state=SEED,n_jobs=-1),
            "Random Forest (20 trees)": lambda: RandomForestRegressor(n_estimators=20,max_depth=12,min_samples_leaf=20,max_features="sqrt",random_state=SEED,n_jobs=-1),
        }, "scikit-learn"
    except ImportError:
        return {
            "Linear Regression": lambda: NumpyLinearRegression(),
            "Decision Tree (depth 8)": lambda: NumpyRandomForestRegressor(n_estimators=1,max_depth=8,min_samples_leaf=30,max_thresholds=8,random_state=SEED),
            "Random Forest (12 trees)": lambda: NumpyRandomForestRegressor(n_estimators=12,max_depth=8,min_samples_leaf=30,max_thresholds=8,random_state=SEED),
            "Random Forest (20 trees)": lambda: NumpyRandomForestRegressor(n_estimators=20,max_depth=12,min_samples_leaf=20,max_thresholds=8,random_state=SEED),
        }, "NumPy fallback (scikit-learn unavailable)"

def sample_rows(frame, limit, seed):
    return frame.sample(n=min(limit,len(frame)),random_state=seed).sort_values("order_date").reset_index(drop=True) if len(frame)>limit else frame.reset_index(drop=True)

def temporal_cv(train, factories):
    dates=np.sort(train.order_date.dt.normalize().unique()); folds=[]
    # Three expanding-window folds. Validation always follows its training dates.
    for frac in (.60,.72,.84):
        cut=max(1,min(len(dates)-1,int(len(dates)*frac)))
        a=train[train.order_date.dt.normalize()<dates[cut]]
        b=train[train.order_date.dt.normalize()>=dates[cut]]
        if len(a) and len(b): folds.append((sample_rows(a,5000,SEED+cut),sample_rows(b,3500,SEED+cut+100)))
    rows=[]
    for name,factory in factories.items():
        for fold,(a,b) in enumerate(folds,1):
            xa,xb,_=encode_fit(a,b); model=factory(); model.fit(xa,a[TARGET].to_numpy(float)); pred=model.predict(xb)
            m=regression_metrics(b[TARGET].to_numpy(float),pred)
            rows.append({"model":name,"fold":fold,"train_rows":len(a),"validation_rows":len(b),**m})
    table=pd.DataFrame(rows)
    return table, table.groupby("model")["RMSE"].mean().sort_values()

def fit_final(name, factories, train, test):
    # Bound the custom NumPy tree fallback cost while retaining a reproducible,
    # distribution-preserving training sample. sklearn uses all training data.
    implementation = make_models()[1]
    fit_train = train if implementation=="scikit-learn" or name=="Linear Regression" else sample_rows(train,15000,SEED)
    xa,xb,features=encode_fit(fit_train,test)
    model=factories[name](); model.fit(xa,fit_train[TARGET].to_numpy(float)); pred=model.predict(xb)
    return model,pred,features,len(fit_train)

def make_charts(y,preds,metrics,triage):
    FIGURES.mkdir(exist_ok=True); font=ImageFont.load_default(); ink="#17324D"; blue="#2878B5"; orange="#E07A32"; green="#4A9B78"
    im=Image.new("RGB",(1000,540),"white"); d=ImageDraw.Draw(im); d.text((30,20),"Chronological test performance (lower MAE/RMSE is better)",fill=ink,font=font)
    names=list(metrics); maxv=max(metrics[n]["RMSE"] for n in names)*1.15
    for i,n in enumerate(names):
        yy=75+i*110; d.text((35,yy),n,fill=ink,font=font)
        for j,k in enumerate(["MAE","RMSE"]):
            val=metrics[n][k]; y0=yy+22+j*32; d.rectangle((220,y0,220+620*val/maxv,y0+22),fill=[blue,orange][j]); d.text((850,y0),f"{k} {val:.2f}",fill=ink,font=font)
        d.text((220,yy+83),f"R² {metrics[n]['R2']:.3f}",fill=ink,font=font)
    im.save(FIGURES/"week4_model_comparison.png")
    im=Image.new("RGB",(900,610),"white"); d=ImageDraw.Draw(im); d.text((25,20),"Selected model: actual versus predicted on chronological test",fill=ink,font=font)
    rng=np.random.default_rng(SEED); ix=rng.choice(len(y),min(3000,len(y)),replace=False); lo=min(float(np.min(y)),float(np.min(preds))); hi=max(float(np.max(y)),float(np.max(preds))); L,T,R,B=90,70,850,520
    d.line((L,B,R,T),fill="#999999",width=2)
    for z in ix:
        x=L+(y[z]-lo)/(hi-lo)*(R-L); yy=B-(preds[z]-lo)/(hi-lo)*(B-T); d.ellipse((x-2,yy-2,x+2,yy+2),fill=blue)
    d.text((L,B+20),f"{lo:.0f} min",fill=ink,font=font); d.text((R-50,B+20),f"{hi:.0f} min",fill=ink,font=font); im.save(FIGURES/"week4_actual_vs_predicted.png")
    labels=["Random baseline","Model triage"]; vals=[triage["random_alerted_long_count"],triage["model_alerted_long_count"]]; top=max(vals)*1.25 or 1
    im=Image.new("RGB",(850,450),"white"); d=ImageDraw.Draw(im); d.text((25,20),"Long-duration cases captured at equal 20% review capacity",fill=ink,font=font)
    for i,(lab,v,col) in enumerate(zip(labels,vals,[orange,green])):
        x=170+i*330; yy=370-v/top*270; d.rectangle((x,yy,x+160,370),fill=col); d.text((x,yy-22),str(v),fill=ink,font=font); d.text((x-20,390),lab,fill=ink,font=font)
    im.save(FIGURES/"week4_triage_optimization.png")

def run():
    REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    data=load_data(); train,test,cutoff=split_by_date(data)
    factories,implementation=make_models()
    cv,ranking=temporal_cv(train,factories); cv.to_csv(REPORTS/"week4_cross_validation.csv",index=False)
    best=ranking.index[0]
    # Evaluate candidates on one shared held-out future period for comparison.
    metrics={}; test_preds={}; trained={}; feature_names=None; sample_counts={}
    for name in factories:
        model,pred,features,nfit=fit_final(name,factories,train,test); test_preds[name]=pred; metrics[name]=regression_metrics(test[TARGET].to_numpy(float),pred); trained[name]=model; sample_counts[name]=nfit; feature_names=features
    selected=best; y=test[TARGET].to_numpy(float); pred=test_preds[selected]
    # Equal-capacity exception review: train-period upper-quartile threshold;
    # alert top 20% future orders by predicted duration versus fixed-seed random.
    threshold=float(train[TARGET].quantile(.75)); actual_long=y>=threshold; k=max(1,int(.20*len(test)))
    model_ix=np.argsort(pred)[-k:]; random_ix=np.random.default_rng(SEED).choice(len(test),k,replace=False)
    triage={"objective":"maximize long-duration test records surfaced to limited review capacity",
      "review_capacity_fraction":.20,"review_slots":k,"long_duration_threshold_min_train_q75":threshold,
      "test_long_duration_count":int(actual_long.sum()),
      "model_alerted_long_count":int(actual_long[model_ix].sum()),"random_alerted_long_count":int(actual_long[random_ix].sum()),
      "model_precision":float(actual_long[model_ix].mean()),"random_precision":float(actual_long[random_ix].mean()),
      "model_recall":float(actual_long[model_ix].sum()/max(1,actual_long.sum())),
      "random_recall":float(actual_long[random_ix].sum()/max(1,actual_long.sum())),
      "interpretation":"Triage only. Alerts do not change delivery outcomes; this is a test-set resource-allocation simulation, not measured time savings."}
    make_charts(y,pred,metrics,triage)
    comp=pd.DataFrame(metrics).T.rename_axis("model").reset_index(); comp.to_csv(REPORTS/"week4_model_comparison.csv",index=False)
    cvmeans=cv.groupby("model").agg(cv_rmse_mean=("RMSE","mean"),cv_rmse_sd=("RMSE","std"),cv_mae_mean=("MAE","mean"),cv_r2_mean=("R2","mean")).reset_index(); cvmeans.to_csv(REPORTS/"week4_cv_summary.csv",index=False)
    pd.DataFrame({"actual_min":y,"predicted_min":pred,"long_duration_actual":actual_long,"model_alert":np.isin(np.arange(len(y)),model_ix)}).to_csv(REPORTS/"week4_test_predictions.csv",index=False)
    (REPORTS/"week4_optimization_results.json").write_text(json.dumps(triage,indent=2),encoding="utf-8")
    result={"dataset":{"path":"data/cleaned_logistics_data.csv","source_note":"Existing cleaned dataset; provenance caveat in data/DATASET_SOURCE.md","rows_used":len(data),"date_min":str(data.order_date.min().date()),"date_max":str(data.order_date.max().date()),"distinct_dates":int(data.order_date.dt.normalize().nunique()),"missing_target_removed":0},
      "problem":{"target":TARGET,"features":NUMERIC+CATEGORICAL,"excluded":"IDs, coordinates, raw timestamps, post-outcome ratings, pickup times and any target-derived fields"},
      "split":{"type":"chronological","cutoff_exclusive_train":str(cutoff.date()),"train_rows":len(train),"test_rows":len(test),"seed":SEED},
      "implementation":implementation,"training_rows_by_model":sample_counts,"cv_mean_rmse":ranking.to_dict(),"selected_model":selected,
      "test_metrics":metrics,"optimization":triage}
    (REPORTS/"week4_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    # Transparent execution summary for notebook/report assembly.
    print(json.dumps(result,indent=2))
    return result,cv,comp,cvmeans,pred,y

if __name__=="__main__": run()
