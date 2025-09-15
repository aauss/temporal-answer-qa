from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from temp_answer_qa import LastToken
from temp_answer_qa.measure_error import tot_measure_error, ttqa_measure_error
from temp_answer_qa.metrics import calculate_metrics
from temp_answer_qa.response_processing import tot_process_response, ttqa_process_response


def eval_tot(results_folder: Path, last_token: LastToken, output_folder: Path):
    files = glob(str(results_folder / f"tot*_{last_token.value}.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {results_folder} for {last_token.value}")
    response_dfs = []
    for file in tqdm(files, desc="Evaluating ToT model responses"):
        file_name_split = Path(file).name.split("_")
        model = file_name_split[2]
        prompting = file_name_split[3]
        df = (
            pd.read_csv(file)
            .assign(model=model, prompting=prompting)
            .pipe(tot_process_response, last_token)
            .pipe(tot_measure_error)
        )
        response_dfs.append(df)
    response_df = pd.concat(response_dfs)
    response_df = _reindex_response_df(response_df)
    response_df = response_df.groupby("split", group_keys=False).apply(calculate_metrics)
    # sMAPE not defined for dates
    response_df.loc[
        response_df.loc[:, "answer_temporal_unit"] == "date",
        "symmetric_absolute_percentage_error",
    ] = None
    response_df.to_csv(output_folder / f"tot_{last_token.value}_evaluated.csv", index=False)


def _reindex_response_df(response_df: pd.DataFrame):
    cols = [
        "question",
        "label",
        "question_type",
        "question_wo_instruct",
        "instruction",
        "answer_format",
        "answer_temporal_unit",
        "split",
        "prompting",
        "model",
    ]
    ref_index = (
        pd.read_pickle(Path(__file__).parent.parent / "data/responses_evaluated/ref_index.pickle").set_index(cols).index
    )
    tmp = response_df.set_index(cols)
    response_df_sorted = tmp.reindex(ref_index)
    response_df_sorted = response_df_sorted.reset_index()
    return response_df_sorted


def eval_ttqa(results_folder: Path, last_token: LastToken, output_folder: Path):
    files = glob(str(results_folder / f"ttqa*_{last_token.value}.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {results_folder} for {last_token.value}")
    response_dfs = []
    for file in tqdm(files, desc="Evaluating TTQA model responses"):
        file_name_split = Path(file).name.split("_")
        model = file_name_split[2]
        prompting = file_name_split[3]
        df = (
            pd.read_csv(file)
            .assign(model=model, prompting=prompting)
            .pipe(ttqa_process_response)
            .pipe(ttqa_measure_error)
        )
        response_dfs.append(df)
    response_df = pd.concat(response_dfs)
    response_df = response_df.pipe(calculate_metrics)
    # sMAPE not defined for dates
    response_df.loc[
        response_df.loc[:, "answer_temporal_unit"].str.contains("date"),
        "symmetric_absolute_percentage_error",
    ] = None
    response_df.loc[
        response_df.loc[:, "answer_temporal_unit"].str.contains("time"),
        "symmetric_absolute_percentage_error",
    ] = None
    response_df.to_csv(output_folder / f"ttqa_{last_token.value}_evaluated.csv", index=False)
