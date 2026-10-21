"""Week 3 descriptive EDA on a reproducible, explicitly synthetic dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Pillow fallback for restricted/offline environments
    plt = None
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/week3_hypothetical_logistics.csv"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "visualizations"
SEED = 20261021

def generate_dataset(path=DATA):
    """Generate one year of simulated urban shipments; no company data is used."""
    rng = np.random.default_rng(SEED)
    n = 1000
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    traffic = rng.choice(["Low", "Moderate", "High", "Severe"], n, p=[.25, .40, .27, .08])
    vehicle = rng.choice(["Motorcycle", "Van", "Truck"], n, p=[.48, .37, .15])
    distance = np.round(np.clip(rng.gamma(2.2, 5.0, n) + 1, 1, 55), 2)
    weight = np.round(np.clip(rng.lognormal(2.2, .75, n), .3, 180), 2)
    traffic_effect = pd.Series(traffic).map({"Low": 0, "Moderate": 6, "High": 15, "Severe": 27}).to_numpy()
    speed = pd.Series(vehicle).map({"Motorcycle": 34, "Van": 29, "Truck": 24}).to_numpy()
    duration = np.maximum(8, 12 + distance / speed * 60 + traffic_effect + .07 * weight + rng.normal(0, 9, n))
    rate = pd.Series(vehicle).map({"Motorcycle": 18, "Van": 25, "Truck": 34}).to_numpy()
    cost = np.maximum(45, 35 + rate * distance + .9 * weight + rng.normal(0, 70, n))
    late_prob = np.clip(.08 + traffic_effect / 100 + (duration > 55) * .20, .03, .80)
    delayed = rng.random(n) < late_prob
    status = np.where(delayed, "Delayed", "On time")
    zones = np.array(["North", "South", "East", "West", "Central"])
    df = pd.DataFrame({
        "shipment_id": [f"S{i:04d}" for i in range(1, n+1)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "origin_zone": rng.choice(zones, n), "destination_zone": rng.choice(zones, n),
        "distance_km": distance, "weight_kg": weight, "traffic_condition": traffic,
        "vehicle_type": vehicle, "delivery_time_min": np.round(duration, 1),
        "transportation_cost": np.round(cost, 2), "shipment_status": status,
        "delayed": delayed,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def run():
    REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    nums = ["distance_km", "weight_kg", "delivery_time_min", "transportation_cost"]
    stats = df[nums].describe(percentiles=[.25,.5,.75]).T
    stats.to_csv(REPORTS / "week3_summary_statistics.csv")
    corr = df[nums].corr()
    corr.to_csv(REPORTS / "week3_correlations.csv")
    group = {}
    for col in ["traffic_condition", "vehicle_type", "shipment_status"]:
        group[col] = df.groupby(col, observed=True).agg(
            shipment_count=("shipment_id", "size"),
            mean_delivery_time_min=("delivery_time_min", "mean"),
            median_delivery_time_min=("delivery_time_min", "median"),
            mean_transportation_cost=("transportation_cost", "mean"),
            delay_rate=("delayed", "mean"),
        ).round(3).to_dict(orient="index")
    monthly = df.groupby(df.order_date.dt.to_period("M")).agg(
        shipments=("shipment_id", "size"), mean_delivery_time_min=("delivery_time_min", "mean"),
        mean_transportation_cost=("transportation_cost", "mean"), delay_rate=("delayed", "mean"))
    monthly.index = monthly.index.astype(str); monthly = monthly.round(3)
    monthly.to_csv(REPORTS / "week3_monthly_trends.csv")
    diagnostics = {"shape": list(df.shape), "dtypes": {k:str(v) for k,v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(), "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_shipment_ids": int(df.shipment_id.duplicated().sum()),
        "seed": SEED, "synthetic": True, "assumptions": {
        "records":1000,"period":"2025 calendar year","distance_km":"clipped gamma-like urban distance, 1–55",
        "weight_kg":"clipped lognormal, 0.3–180","delivery_time":"distance, vehicle speed, traffic, weight and random variation",
        "cost":"distance and vehicle base rate plus weight and random variation",
        "delay":"probabilistic indicator informed by traffic and simulated duration"}}
    (REPORTS / "week3_data_quality.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (REPORTS / "week3_group_comparisons.json").write_text(json.dumps(group, indent=2), encoding="utf-8")
    if plt is None:
        make_pillow_charts(df, corr, monthly)
    else:
        make_matplotlib_charts(df, corr, monthly)
    results={"shape":diagnostics["shape"],"missing":diagnostics["missing_values"],"duplicates":diagnostics["duplicate_rows"],"stats":stats.round(3).to_dict(orient="index"),"correlations":corr.round(3).to_dict(),"groups":group,"monthly":monthly.to_dict(orient="index"),"delay_rate":round(float(df.delayed.mean()),3)}
    (REPORTS/"week3_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))
    return df, results

def make_matplotlib_charts(df, corr, monthly):
    plt.style.use("seaborn-v0_8-whitegrid")
    def save(name):
        plt.tight_layout(); plt.savefig(FIGURES / name, dpi=160, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8,5)); plt.hist(df.delivery_time_min, bins=28, color="#2878B5", edgecolor="white"); plt.xlabel("Delivery time (minutes)"); plt.ylabel("Shipments"); plt.title("Synthetic shipment delivery-time distribution"); save("week3_delivery_time_distribution.png")
    plt.figure(figsize=(8,5)); plt.hist(df.transportation_cost, bins=28, color="#E07A35", edgecolor="white"); plt.xlabel("Transportation cost (simulated currency units)"); plt.ylabel("Shipments"); plt.title("Synthetic transportation-cost distribution"); save("week3_transportation_cost_distribution.png")
    plt.figure(figsize=(8,5)); plt.scatter(df.distance_km, df.delivery_time_min, alpha=.45, s=18, color="#2878B5"); z=np.polyfit(df.distance_km,df.delivery_time_min,1); x=np.linspace(df.distance_km.min(),df.distance_km.max(),100); plt.plot(x,np.polyval(z,x),color="#C23B22",label="Descriptive linear fit"); plt.xlabel("Distance (km)"); plt.ylabel("Delivery time (minutes)"); plt.title("Delivery time versus distance"); plt.legend(); save("week3_delivery_time_vs_distance.png")
    fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(corr, vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(nums)),nums,rotation=35,ha="right"); ax.set_yticks(range(len(nums)),nums); ax.set_title("Correlation among numeric shipment variables");
    for i in range(len(nums)):
        for j in range(len(nums)): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="Pearson correlation"); save("week3_correlation_heatmap.png")
    order=["Low","Moderate","High","Severe"]; g=df.groupby("traffic_condition").delivery_time_min.agg(["mean","std"]).reindex(order)
    plt.figure(figsize=(8,5)); plt.bar(g.index,g["mean"],yerr=g["std"],capsize=4,color="#4A9B78"); plt.xlabel("Traffic condition"); plt.ylabel("Mean delivery time (minutes; error bars = SD)"); plt.title("Delivery time by traffic condition"); save("week3_delivery_time_by_traffic.png")
    g=df.groupby("vehicle_type").transportation_cost.agg(["mean","std"]).reindex(["Motorcycle","Van","Truck"])
    plt.figure(figsize=(8,5)); plt.bar(g.index,g["mean"],yerr=g["std"],capsize=4,color="#8064A2"); plt.xlabel("Vehicle type"); plt.ylabel("Mean transportation cost (simulated units; error bars = SD)"); plt.title("Transportation cost by vehicle type"); save("week3_cost_by_vehicle.png")
    g=df.groupby("shipment_status").delivery_time_min.agg(["mean","std"]).reindex(["On time","Delayed"])
    plt.figure(figsize=(7,5)); plt.bar(g.index,g["mean"],yerr=g["std"],capsize=4,color=["#4A9B78","#C23B22"]); plt.xlabel("Shipment status"); plt.ylabel("Mean delivery time (minutes; error bars = SD)"); plt.title("Delivery time by shipment status"); save("week3_delivery_time_by_status.png")
    plt.figure(figsize=(10,5)); plt.plot(monthly.index,monthly.mean_delivery_time_min,marker="o",label="Mean delivery time (min)"); plt.ylabel("Mean delivery time (minutes)"); plt.xlabel("Month in 2025"); plt.xticks(rotation=45); plt.title("Monthly simulated delivery-time trend"); plt.legend(); save("week3_monthly_delivery_time.png")
def make_pillow_charts(df, corr, monthly):
    """Readable PNG charts using Pillow when matplotlib is unavailable."""
    W,H=1000,620; blue=(40,120,181); orange=(224,122,53); dark=(35,45,55)
    def frame(title,xlab,ylab):
        im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im); d.text((W//2-200,25),title,fill=dark); d.line((90,80,90,530),fill=dark,width=2); d.line((90,530,940,530),fill=dark,width=2); d.text((420,570),xlab,fill=dark); d.text((15,290),ylab,fill=dark); return im,d
    def bars(title, labels, values, xlab, ylab, filename, color=blue):
        im,d=frame(title,xlab,ylab); top=max(values)*1.15 if max(values)>0 else 1; bw=700/max(len(values),1)
        for i,(lab,val) in enumerate(zip(labels,values)):
            x=120+i*bw; y=520-val/top*400; d.rectangle((x,y,x+bw*.65,520),fill=color); d.text((x,530),str(lab),fill=dark); d.text((x,y-22),f"{val:.1f}",fill=dark)
        im.save(FIGURES/filename)
    def hist(series,title,xlab,filename,color):
        bins=np.histogram(series,bins=24); labels=bins[1]; vals=bins[0]; im,d=frame(title,xlab,"Shipments"); top=max(vals)*1.1; bw=820/len(vals)
        for i,v in enumerate(vals):
            x=100+i*bw; y=520-v/top*400; d.rectangle((x,y,x+bw-2,520),fill=color)
        im.save(FIGURES/filename)
    hist(df.delivery_time_min,"Synthetic shipment delivery-time distribution","Delivery time (minutes)","week3_delivery_time_distribution.png",blue)
    hist(df.transportation_cost,"Synthetic transportation-cost distribution","Cost (simulated currency units)","week3_transportation_cost_distribution.png",orange)
    im,d=frame("Delivery time versus distance","Distance (km)","Delivery time (minutes)");
    for x,y in zip(df.distance_km,df.delivery_time_min): d.ellipse((95+x/55*820,515-y/180*400,99+x/55*820,519-y/180*400),fill=blue)
    im.save(FIGURES/"week3_delivery_time_vs_distance.png")
    im=Image.new("RGB",(850,700),"white"); d=ImageDraw.Draw(im); d.text((280,20),"Correlation among numeric shipment variables",fill=dark); cols=list(corr.columns); palette=[(178,24,43),(244,165,130),(247,247,247),(146,197,222),(33,102,172)]
    for i,row in enumerate(cols):
        d.text((5,150+i*110),row,fill=dark); d.text((170+i*145,105),row,fill=dark)
        for j,col in enumerate(cols):
            v=float(corr.loc[row,col]); ix=min(4,max(0,int(round((v+1)*2)))); x=170+j*145; y=140+i*110; d.rectangle((x,y,x+130,y+95),fill=palette[ix]); d.text((x+50,y+38),f"{v:.2f}",fill=dark)
    im.save(FIGURES/"week3_correlation_heatmap.png")
    g=df.groupby("traffic_condition").delivery_time_min.mean().reindex(["Low","Moderate","High","Severe"]); bars("Delivery time by traffic condition",g.index,g.values,"Traffic condition","Mean delivery time (minutes)","week3_delivery_time_by_traffic.png",(74,155,120))
    g=df.groupby("vehicle_type").transportation_cost.mean().reindex(["Motorcycle","Van","Truck"]); bars("Transportation cost by vehicle type",g.index,g.values,"Vehicle type","Mean transportation cost (units)","week3_cost_by_vehicle.png",(128,100,162))
    g=df.groupby("shipment_status").delivery_time_min.mean().reindex(["On time","Delayed"]); bars("Delivery time by shipment status",g.index,g.values,"Shipment status","Mean delivery time (minutes)","week3_delivery_time_by_status.png",orange)
    bars("Monthly simulated delivery-time trend",monthly.index,monthly.mean_delivery_time_min.values,"Month in 2025","Mean delivery time (minutes)","week3_monthly_delivery_time.png",blue)

if __name__ == "__main__":
    if not DATA.exists(): generate_dataset()
    run()
