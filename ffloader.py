import io
import zipfile
import requests
import pandas as pd
from pathlib import Path


class FamaFrenchLoader:
    BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

    DATASETS = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum": "F-F_Momentum_Factor_CSV.zip",
    "bm_6": "6_Portfolios_2x3_CSV.zip",
    "bm_25": "25_Portfolios_5x5_CSV.zip",
    "ep_6": "6_Portfolios_ME_EP_2x3_CSV.zip",
    "ep_25": "25_Portfolios_ME_EP_5x5_CSV.zip",
    "beta": "25_Portfolios_ME_BETA_5x5_CSV.zip",
    "ff49": "49_Industry_Portfolios_CSV.zip",
    "ff49_daily": "49_Industry_Portfolios_daily_CSV.zip",
    "inv6": "6_Portfolios_ME_INV_2x3_CSV.zip"}

    def __init__(self, cache_dir="macro_data/fama_french"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset: str, force_download: bool = False, freq: str = "monthly") -> pd.DataFrame:
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset. Choose from: {list(self.DATASETS)}")

        cache_file = self.cache_dir / f"{dataset}.csv"

        if cache_file.exists() and not force_download:
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        raw_text = self._download_zip(self.DATASETS[dataset])
        df = self._parse_table(raw_text, freq=freq)

        df.to_csv(cache_file)
        return df
    
    def _download_zip(self, filename: str) -> str:
        url = f"{self.BASE_URL}/{filename}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                return f.read().decode("latin1")

    def _parse_table(self, raw_text: str, freq: str = "monthly") -> pd.DataFrame:
        lines = raw_text.splitlines()

        start = None
        end = None

        for i, line in enumerate(lines):
            if line.startswith(","):
                start = i
                break

        if start is None:
            raise ValueError("Could not find table header.")

        expected_len = 6 if freq == "monthly" else 8
        date_format = "%Y%m" if freq == "monthly" else "%Y%m%d"

        for i in range(start + 1, len(lines)):
            first_col = lines[i].split(",")[0].strip()

            if not first_col.isdigit() or len(first_col) != expected_len:
                end = i
                break

        csv_text = "\n".join(lines[start:end])

        df = pd.read_csv(io.StringIO(csv_text), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str), format=date_format)

        if freq == "monthly":
            df.index = df.index + pd.offsets.MonthEnd(0)

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df / 100

        return df



#Example usage:

# loader = FamaFrenchLoader()

# ff5 = loader.load("ff5")
# ep6 = loader.load("ep_6")
# bm6 = loader.load("bm_6")
# momentum = loader.load("momentum")
