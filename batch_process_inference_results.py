import json
import os
import platform
import subprocess
from math import ceil
from multiprocessing import Process
from pathlib import Path

import pandas

combined_csv_name = "combined-data.csv"
calculated_avgs_file_name = "calculated_averages.json"
r_executable = "C:\\Program Files\\R\\R-4.2.2\\bin\\x64\\Rscript.exe" if platform.system() == 'Windows' else "Rscript"


def calculate_all_avgs(eval_dir: str):
    for size_func in [
        "pixels",
        "edges-fine",
        "edges-coarse",
        "nodes",
        "chars",
    ]:
        calculate_avgs(eval_dir=eval_dir, size_func=size_func)


def calculate_avgs(eval_dir: str, size_func: str):
    combined_csv_loc = os.path.join(eval_dir, size_func, combined_csv_name)
    df = pandas.read_csv(combined_csv_loc)
    stats = {
        "avg_bcuded_f1": df["bcubed.f1"].mean(),
        "avg_bcuded_recall": df["bcubed.recall"].mean(),
        "avg_bcuded_precision": df["bcubed.precision"].mean(),
        "avg_bcuded_f1star": 2.0 * df["bcubed.recall"].mean() * df["bcubed.precision"].mean() / (
                df["bcubed.recall"].mean() + df["bcubed.precision"].mean())
    }
    with open(os.path.join(os.path.join(eval_dir, size_func), calculated_avgs_file_name), "w") as handle:
        json.dump(stats, handle, indent=6)


def combine_all_csvs(eval_dir: str):
    for size_func in [
        "pixels",
        "edges-fine",
        "edges-coarse",
        "nodes",
        "chars",
    ]:
        combine_csvs(eval_dir=eval_dir, size_func=size_func)


def combine_csvs(eval_dir: str, size_func: str):
    target_dir = os.path.join(eval_dir, size_func)
    file_names = [d.name for d in os.scandir(target_dir) if
                  d.name.split(".")[-1] in {"json", "csv"}]
    header_line = '"","bcubed.precision","bcubed.recall","bcubed.f1"\n'
    combined_data = ""
    for file_name in file_names:
        full_name = os.path.join(target_dir, file_name)
        with open(full_name, "r") as handle:
            cur_header_line = handle.readline()
            assert cur_header_line == header_line, f"Noticing a different header-line:\n{cur_header_line}"
            cur_data_line = handle.readline()
            assert cur_data_line != "", \
                f"Seeing an empty data line. size_function is {size_func}, file name is {full_name}"
            combined_data += cur_data_line
            assert handle.readline() == '', \
                f"Seeing a file with more than one data line. size_function is {size_func}, file name is {full_name}"
    with open(os.path.join(target_dir, combined_csv_name), "w") as handle:
        handle.write(header_line + combined_data)


def single_evaluation(
        flattened_inferences_dir: str,
        outputs_dir: str,  # Assume this directory is already specific to a size function.
        img_id: str,
        algorithm_segmentation: str,
        combined_dataset_dir: str = "webis-webseg-20-combined",
        evaluation_script_loc: str = "cikm20/src/main/r/evaluate-segmentation.R",
        size_function: str = "pixels"
):
    assert size_function in ["pixels", "edges-fine", "edges-coarse", "nodes", "chars"], \
        'size_function should be one of ["pixels", "edges-fine", "edges-coarse", "nodes", "chars"]'
    args = [
        r_executable, evaluation_script_loc,
        "--algorithm-segmentation", algorithm_segmentation,
        "--algorithm", os.path.join(flattened_inferences_dir, f"{img_id}.json"),
        "--ground-truth", os.path.join(combined_dataset_dir, img_id, "ground-truth.json"),
        '--output', os.path.join(outputs_dir, f"{img_id}.csv"),
        "--size-function", size_function
    ]
    subprocess.check_call(args=args, cwd=os.getcwd())


def single_flatten_segments(
        fitted_inferences_dir: str,
        outputs_dir: str,
        img_id: str,
        flatten_script_loc: str = "cikm20/src/main/r/flatten-segmentations.R"
):
    args = [
        r_executable, flatten_script_loc,
        "--input", os.path.join(fitted_inferences_dir, f"{img_id}.json"),
        '--output', os.path.join(outputs_dir, f"{img_id}.json")
    ]
    subprocess.check_call(args=args, cwd=os.getcwd())


def extract_image_ids(target_dir: str) -> list[str]:
    return [d.name[:6] for d in os.scandir(target_dir) if
            d.name.split(".")[-1] == "json"]


def batch_evaluations(
        flattened_inferences_dir: str,
        outputs_dir: str,
        algorithm_segmentation: str,
        combined_dataset_dir: str = "webis-webseg-20-combined",
        evaluation_script_loc: str = "cikm20/src/main/r/evaluate-segmentation.R",
):
    ids = extract_image_ids(flattened_inferences_dir)

    for size_func in [
        "pixels",
        "edges-fine",
        "edges-coarse",
        "nodes",
        "chars",
    ]:
        print("Evaluating on " + size_func)
        processes = []
        Path(outputs_dir + "/" + size_func).mkdir(parents=True, exist_ok=True)
        for i, img_id in enumerate(ids):
            p = Process(
                target=single_evaluation,
                args=(
                    flattened_inferences_dir,
                    os.path.join(outputs_dir, size_func),
                    img_id,
                    algorithm_segmentation,
                    combined_dataset_dir,
                    evaluation_script_loc,
                    size_func
                ),
            )
            processes.append(p)
        start_and_join_processes(processes=processes)


def batch_flatten_segments(
        fitted_inferences_dir: str,
        outputs_dir: str,
        flatten_script_loc: str = "cikm20/src/main/r/flatten-segmentations.R"
):
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    ids = extract_image_ids(fitted_inferences_dir)
    processes = []
    for i, img_id in enumerate(ids):
        p = Process(
            target=single_flatten_segments,
            args=(
                fitted_inferences_dir,
                outputs_dir,
                img_id,
                flatten_script_loc
            ),
        )
        processes.append(p)
    start_and_join_processes(processes=processes)


def single_fit_segments(
        raw_inferences_dir: str,
        outputs_dir: str,
        img_id: str,
        segmentations_name: str,
        combined_dataset_dir: str = "webis-webseg-20-combined",
        fit_script_loc: str = "cikm20/src/main/r/fit-segmentations-to-dom-nodes.R",
):
    args = [
        r_executable, fit_script_loc,
        "--input", os.path.join(raw_inferences_dir, f"{img_id}.json"),
        "--segmentations", segmentations_name,
        '--nodes', os.path.join(combined_dataset_dir, img_id, "nodes.csv"),
        '--output', os.path.join(outputs_dir, f"{img_id}.json")
    ]
    subprocess.check_call(args=args, cwd=os.getcwd())


def batch_fit_segments(
        raw_inferences_dir: str,
        outputs_dir: str,
        segmentations_name: str,
        combined_dataset_dir: str = "webis-webseg-20-combined",
        fit_script_loc: str = "cikm20/src/main/r/fit-segmentations-to-dom-nodes.R",
):
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    ids = extract_image_ids(raw_inferences_dir)
    processes = []
    for i, img_id in enumerate(ids):
        p = Process(
            target=single_fit_segments,
            args=(
                raw_inferences_dir,
                outputs_dir,
                img_id,
                segmentations_name,
                combined_dataset_dir,
                fit_script_loc,
            ),
        )
        processes.append(p)
    start_and_join_processes(processes=processes)


def start_and_join_processes(processes: list[Process]):
    p_count = len(processes)
    simultaneous_process_numb = 50
    batch_count = ceil(p_count / simultaneous_process_numb)
    for i in range(batch_count):
        batch_end_idx = p_count if i == batch_count - 1 else (i + 1) * simultaneous_process_numb
        batch = processes[i * simultaneous_process_numb:batch_end_idx]
        print(f"Processing the {i + 1}th batch now. There are {batch_count} batches in total.")
        for process in batch:
            process.start()
        for process in batch:
            process.join()
            assert process.exitcode == 0, "Some process exited with none-zero exit code! Please check what is going on!"


def batch_replace(target_dir: str, original: str, new: str):
    file_names = [d.name for d in os.scandir(target_dir) if
                  d.name.split(".")[-1] == "json"]
    for file_name in file_names:
        with open(os.path.join(target_dir, file_name), "r") as handle:
            original_str = handle.read()
        replaced_str = original_str.replace(original, new)
        with open(os.path.join(target_dir, file_name), "w") as handle:
            handle.write(replaced_str)


if __name__ == '__main__':
    algorithm = "htc"
    train_target_type = "screenshots"
    node_fit_segmentation_name = "mmdetection_segms"
    print("Fitting segmentations.")
    batch_fit_segments(
        raw_inferences_dir=f"{algorithm}/inference_out/{train_target_type}/original_inferences",
        outputs_dir=f"{algorithm}/inference_out/{train_target_type}/dom_node_fitted_annotations",
        segmentations_name=node_fit_segmentation_name,
    )
    print("Replacing segmentation names.")
    batch_replace(
        target_dir=f"{algorithm}/inference_out/{train_target_type}/dom_node_fitted_annotations",
        original=f"{node_fit_segmentation_name}.fitted",
        new="generated_segmentation"
    )
    print("Flattening segmentations.")
    batch_flatten_segments(
        fitted_inferences_dir=f"{algorithm}/inference_out/{train_target_type}/dom_node_fitted_annotations",
        outputs_dir=f"{algorithm}/inference_out/{train_target_type}/flattened_annotations",
    )
    print("Evaluating segmentations")
    batch_evaluations(
        flattened_inferences_dir=f"{algorithm}/inference_out/{train_target_type}/flattened_annotations",
        outputs_dir=f"{algorithm}/inference_out/{train_target_type}/evaluations",
        algorithm_segmentation="generated_segmentation"
    )
    print("Combining CSVs")
    combine_all_csvs(eval_dir=f"{algorithm}/inference_out/{train_target_type}/evaluations")
    print("Calculating average statistics.")
    calculate_all_avgs(eval_dir=f"{algorithm}/inference_out/{train_target_type}/evaluations")
