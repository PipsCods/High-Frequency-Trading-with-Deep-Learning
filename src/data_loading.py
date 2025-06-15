# Package imports
import pandas as pd 
import numpy as np
import glob
from tqdm import tqdm

def main(raw_data_path: str =  "../data/raw/high_10m/*.csv.gz", process_data_path: str = "../data/processed/high_10m.parquet"):
    # Take the path of the parent directory of the current file
    files = sorted(glob.glob(str(raw_data_path)))
    days = []
    for file in tqdm(files, desc= "Loading data"):
        day = pd.read_csv(file, compression= "gzip",  parse_dates= ["DATE"])
        days.append(day)

    df = pd.concat(days, ignore_index=True)
    df = df.sort_values(by=["SYMBOL", "DATE"])

    df["RETURN_NoOVERNIGHT"] = (df.groupby(["SYMBOL", "DATE"])["MID_OPEN"].pct_change()) # Best way to calcuate the return
    df["RETURN_NoOVERNIGHT"] = df["RETURN_NoOVERNIGHT"].fillna(0)

    df["LOG_RETURN_NoOVERNIGHT"] = np.log(1 + df["RETURN_NoOVERNIGHT"])
    df["LOG_RETURN_NoOVERNIGHT"] = df["LOG_RETURN_NoOVERNIGHT"].fillna(0)

    # Calculate the return on the column MID_OPEN, for each stock "SYMBOL" 
    df["RETURN_SiOVERNIGHT"] = (df.groupby("SYMBOL")["MID_OPEN"].pct_change()) # Best way to calcuate the return
    df["RETURN_SiOVERNIGHT"] = df["RETURN_SiOVERNIGHT"].fillna(0)

    df["LOG_RETURN_SiOVERNIGHT"] = np.log(1 + df["RETURN_SiOVERNIGHT"])
    df["LOG_RETURN_SiOVERNIGHT"] = df["LOG_RETURN_SiOVERNIGHT"].fillna(0)

    df.to_parquet(process_data_path, index = False)

if __name__ == "__main__":
    main()