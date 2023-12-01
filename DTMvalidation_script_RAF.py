import sys
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=15)

file_paths = sorted(glob("filepath\\*raw.csv", recursive=True))

print(*file_paths,sep="\n")

results = []
for file_path in file_paths: 
    df = pd.read_csv(file_path, header=0)
    df.index = df.index//(df.shape[0]/df.iloc[:,0].dropna().shape[0])
    df.reset_index(inplace=True)
    s1 = df.GPS_height.dropna()
    s2 = df[["index","DEM_height"]].groupby("index").mean().DEM_height
    if len(s1) != len(s2): 
        print(f"{file_path} result different length")
        print(s1,s2)
    else:
        print(f"{file_path} results same length")

    df = pd.DataFrame({"GPS_dist": df.GPS_dist.dropna(), "GPS_height":s1,"DEM_height":s2})
    df.GPS_dist = (df.GPS_dist.max() - df.GPS_dist).values

    res = stats.linregress(df.GPS_height, df.DEM_height)
    r2 = res.rvalue**2

    rmse = np.round(np.sqrt(np.mean((df.GPS_height-df.DEM_height)**2)), 2) #RMSE 
    mae = np.round((df.GPS_height-df.DEM_height).abs().mean(), 2)
    bias = np.round((df.DEM_height-df.GPS_height).mean(), 2)

    transect_string = '_'.join(file_path.split('_')[-3:-1])[5:]
    results.append([int(transect_string.split("_")[1]), int(transect_string.split("_")[0][-1]), r2, rmse, mae, bias])

    print(f"plotting {file_path}")

    fig,axs = plt.subplots(1,2, figsize=(12,6), tight_layout=True) #returns and unpacks figure (picture) and axes (subplots)

    axs[0].plot(df.GPS_dist, df.GPS_height, label="GPS", linewidth=4)
    axs[0].plot(df.GPS_dist, df.DEM_height, label="DEM", linewidth=4)
    axs[0].invert_xaxis()
    axs[0].set_xlabel("distance to the shoreline (m)", fontsize=15)
    axs[0].tick_params(axis="x", labelsize=15)
    axs[0].tick_params(axis="y", labelsize=15)
    axs[0].set_ylabel("elevation (m)", fontsize=15)
    axs[0].set_title("profile comparison", fontsize=20)
    axs[0].legend()

    axs[1].scatter(df.GPS_height, df.DEM_height, s=4, label="elevation")
    axs[1].plot(df.GPS_height, res.intercept + res.slope*df.GPS_height, 'r', linewidth=.5)
    axs[1].plot([],[], linewidth=0, label=f'r2={r2:.2f}\nrmse={rmse:.2f}m\nmae={mae:.2f}m\nbias={bias:.2f}m')
    axs[1].set_xlabel("GPS elevation (m)", fontsize=15)
    axs[1].tick_params(axis="x", labelsize=15)
    axs[1].tick_params(axis="y", labelsize=15)
    axs[1].set_ylabel("DEM elevation (m)", fontsize=15)
    axs[1].set_title("regression", fontsize=20)
    axs[1].legend(fontsize=15)

    fig.suptitle(transect_string, fontsize=30)

    if sys.argv[1]=="show":
        plt.show()
    elif sys.argv[1]=="plot":
        plt.savefig(file_path.replace(".csv","_new.png"), dpi = 300)

    plt.close()

    #break

if sys.argv[1] == "plot":
    columns = ["year","transect","r2","rmse","mae","bias"]
    df = pd.DataFrame(results,columns=columns)
    print(df)
    df.to_csv("filepath\\summary.csv")

