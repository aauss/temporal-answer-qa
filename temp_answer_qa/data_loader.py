import pandas as pd

from temp_answer_qa import DATA_DIR, ToTSplit, TTQASplit


class DataLoader:
    def load_ttqa(self, split: TTQASplit | None = None, test_mode: bool = False) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "questions/ttqa.csv")
        if split:
            df = df.loc[df.loc[:, "split"] == split]
        return df.sample(50, random_state=146) if test_mode else df

    def load_tot(self, split: ToTSplit | None = None, test_mode: bool = False) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "questions/tot.csv")
        if split:
            df = df.loc[df.loc[:, "split"] == split]
        return df.sample(50, random_state=146) if test_mode else df
