import copy
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, asdict
from multiprocessing import Process
from typing import Tuple

import yaml


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/data/"
OUT_METADATA_FOLDER = "/veld/output/metadata/"
TMP_FOLDER = "/tmp/"


@dataclass
class ConfigReading:
    in_folder: str = None
    in_file_path: str = None


@dataclass
class ConfigReadingTxt(ConfigReading):
    pass


@dataclass
class ConfigWriting:
    out_folder: str = None
    out_file_path: str = None


@dataclass
class ConfigWritingTxt(ConfigWriting):
    pass


@dataclass
class ConfigWritingClean(ConfigWriting):
    config_writing_clean: ConfigWriting = None
    config_writing_dirty: ConfigWriting = None


@dataclass
class ConfigWritingMetadata():
    out_metadata_folder: str = None
    out_metadata_file_path: str = None
    out_metadata_description: str = None
    out_metadata_topic: str | list[str] = None
    out_metadata_content: str | list[str] = None


@dataclass
class ConfigProcessing:
    cpu_count: int = None


@dataclass
class ConfigProcessingChangeCase(ConfigProcessing):
    set_case: str = None


@dataclass
class ConfigProcessingRegexReplace(ConfigProcessing):
    regex_sub: Tuple[str, str] = None


@dataclass
class ConfigProcessingClean(ConfigProcessing):
    min_clean_char_percentage: float = None


def get_env_var(var_name, cast_func=None):
    var_content = os.getenv(var_name)
    if var_content:
        print(f"{var_name}: {var_content}")
    if cast_func:
        try:
            var_content = cast_func(var_content)
        except:
            raise Exception(f"Could not convert var '{var_name}' to {cast_func}")
    return var_content


def get_env_var_to_list(var_name):
    var_content = get_env_var(var_name)
    var_content_list = var_content.split(",")
    return var_content_list


def get_processing_func_name():
    processing_func_name = sys.argv[1]
    return processing_func_name


def concatenate_folder_and_file(folder, file):
    if file:
        return folder + file
    else:
        return None


def get_config_reading():
    config_reading = ConfigReading(
        in_folder=IN_FOLDER,
        in_file_path=concatenate_folder_and_file(IN_FOLDER, get_env_var("in_file")),
    )
    return config_reading


def get_config_writing():
    processing_func_name = get_processing_func_name()
    if processing_func_name == "clean":
        config_writing = ConfigWritingClean(
            config_writing_clean=ConfigWriting(
                out_folder=OUT_FOLDER,
                out_file_path=concatenate_folder_and_file(OUT_FOLDER, get_env_var("out_file_clean")),
            ),
            config_writing_dirty=ConfigWriting(
                out_folder=OUT_FOLDER,
                out_file_path=concatenate_folder_and_file(OUT_FOLDER, get_env_var("out_file_dirty")),
            ),
        )
    else:
        config_writing = ConfigWriting(
            out_folder=OUT_FOLDER,
            out_file_path=concatenate_folder_and_file(OUT_FOLDER, get_env_var("out_file")),
        )
    return config_writing


def get_config_writing_metadata():
    out_metadata_file = get_env_var("out_metadata_file")
    if out_metadata_file:
        out_metadata_file_path = OUT_METADATA_FOLDER + out_metadata_file
    else:
        out_metadata_file = None
    config_writing_metadata = ConfigWritingMetadata(
        out_metadata_folder=OUT_METADATA_FOLDER,
        out_metadata_file_path=out_metadata_file_path,
        out_metadata_description=get_env_var("out_metadata_description"),
        out_metadata_topic=get_env_var_to_list("out_metadata_topic"),
        out_metadata_content=get_env_var_to_list("out_metadata_content"),
    )
    return config_writing_metadata


def get_config_processing():
    processing_func_name = get_processing_func_name()
    print(f"processing: {processing_func_name}")
    config_processing = ConfigProcessing(
        cpu_count=get_env_var("cpu_count", int),
    )
    if processing_func_name == "change_case":
        config_processing = ConfigProcessingChangeCase(
            **asdict(config_processing),
            set_case=get_env_var("set_case"),
        )
    elif processing_func_name == "remove_punctuation":
        config_processing = ConfigProcessingRegexReplace(
            **asdict(config_processing),
            regex_sub=(r"[^\w\s]", "")
        )
    # TODO: create veld service for regex_replace
    elif processing_func_name == "regex_replace": 
        regex_pattern_match = get_env_var("regex_pattern_match")
        regex_pattern_replacement = get_env_var("regex_pattern_replacement")
        config_processing = ConfigProcessingRegexReplace(
            **asdict(config_processing),
            regex_sub=(regex_pattern_match, regex_pattern_replacement),
        )
    elif processing_func_name == "clean":
        config_processing = ConfigProcessingClean(
            **asdict(config_processing),
            min_clean_char_percentage=get_env_var("min_clean_char_percentage", float),
        )
    return config_processing


def get_filetype_of_config(config_reading_or_writing):
    if type(config_reading_or_writing) in [ConfigReadingTxt, ConfigWritingTxt]:
        return "txt"


def get_func_reading(config_reading):
    if type(config_reading) is ConfigReadingTxt:
        return func_reading_txt


def get_func_writing(config_writing):
    if type(config_writing) is ConfigWritingTxt:
        return func_writing_txt


def get_func_processing(config_processing):
    if type(config_processing) is ConfigProcessingChangeCase:
        return func_processing_change_case
    elif type(config_processing) is ConfigProcessingClean:
        return func_processing_clean
    elif type(config_processing) is ConfigProcessingRegexReplace:
        return func_processing_regex_replace


def func_reading_txt(config_reading, f_in, segment_start_end_list=None):
    for i_line, line in enumerate(f_in):
        if segment_start_end_list:
            if i_line >= segment_start_end_list[0]:
                if i_line < segment_start_end_list[1]:
                    yield (i_line, line)
                else:
                    break
        else:
            yield (i_line, line)


def func_writing_txt(config_writing, text, f_out):
    f_out.write(text)


def func_processing_change_case(config_processing, text):
    if config_processing.set_case == "upper":
        func_case = str.upper
    elif config_processing.set_case == "lower":
        func_case = str.lower
    text_processed = func_case(text)
    return text_processed


def func_processing_clean(config_processing, text):
    count_clean = 0
    count_dirty = 0
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("L") or cat.startswith("Z"):
            count_clean += 1
        else:
            count_dirty += 1
    percentage_clean_char = (100 * count_clean)  / (count_clean + count_dirty)
    if percentage_clean_char >= config_processing.min_clean_char_percentage:
        return (text, True)
    else:
        return (text, False)


def func_processing_regex_replace(config_processing, text):
    text_processed = re.sub(config_processing.regex_sub)
    return text_processed


def count_texts_of_output_individual(config_writing, file_name_pattern=None):
    num_lines = 0
    if config_writing.out_file_path:
        with open(config_writing.out_file_path, "r") as f:
            num_lines = count_texts_in_file(config_writing, f)
    else:
        num_lines = 0
        for file in os.listdir(config_writing.out_folder):
            config_writing.out_file_path = config_writing.out_folder + file
            if not file_name_pattern or file_name_pattern in file:
                with open(config_writing.out_file_path, "r") as f:
                    num_lines += count_texts_in_file(config_writing, f)
    return num_lines


def count_texts_of_output_main(config_writing):
    num_lines = 0
    if type(config_writing) is ConfigWritingClean:
        num_lines = count_texts_of_output_individual(config_writing.config_writing_clean, "_clean")
    else:
        num_lines = count_texts_of_output_individual(config_writing)
    return num_lines


def write_veld_data_yaml(config_writing_metadata, config_writing):
    if type(config_writing) is ConfigWritingClean:
        out_folder = config_writing.config_writing_clean.out_folder
    else:
        out_folder = config_writing.out_folder
    result = subprocess.run(["du", "-sh", out_folder], capture_output=True, text=True)
    data_size = result.stdout.split()[0]
    num_lines = count_texts_of_output_main(config_writing)
    # if type(config_writing) is ConfigWritingTxt:
    #     file_type = "txt"
    veld_data_yaml = {
        "x-veld": {
            "data": {
                "description": config_writing.out_metadata_description,
                "topic": config_writing.out_metadata_topic,
                "content": config_writing.out_metadata_content,
                "file_type": file_type,
                "additional": {
                    "data size": data_size,
                    "number of lines": num_lines,
                }
            }
        }
    }
    with open(config_writing.out_metadata_file_path, "w") as f:
        yaml.dump(veld_data_yaml, f, sort_keys=False)


def merge_tmp_individual(config_writing, file_name_pattern):
    with open(config_writing.out_file_path, "w") as f_out:
        tmp_file_path_list = sorted([TMP_FOLDER + "/" + f for f in os.listdir(TMP_FOLDER)])
        for tmp_file_path in tmp_file_path_list:
            if not file_name_pattern or file_name_pattern in tmp_file_path:
                with open(tmp_file_path, "r") as f_in:
                    for line in f_in:
                        f_out.write(line)


def merge_tmp_main(config_writing):
    print("joining tmp files into one.")
    if type(config_writing) is ConfigWritingClean:
        merge_tmp_individual(config_writing.config_writing_clean, "clean") 
        merge_tmp_individual(config_writing.config_writing_dirty, "dirty") 
    else:
        merge_tmp_individual(config_writing)


def create_segment_start_end_list_of_quantity(num_total, num_segments):
    segment_start_end_list = []
    step = num_total / num_segments
    i_start = 0
    for i_segment in range(1, num_segments + 1):
        if i_segment < num_segments:
            i_end = round(i_segment * step)
            segment_start_end_list.append((i_start, i_end))
            i_start = i_end
        else:
            segment_start_end_list.append((i_start, num_total))
    return segment_start_end_list


def count_texts_in_file(config_reading, f_in):
    func_reading = get_func_reading(config_reading)
    with open(config_reading.in_file_path, "r") as f_in:
        func_reading = get_func_reading(config_reading)
        i_line = 0
        for i_line, _ in func_reading(config_reading, f_in):
            pass
    num_lines = i_line + 1
    return num_lines


def create_segment_start_end_list_of_file(config_reading, num_segments):
    print("- creating index segments of file -----------------------------------")
    segment_start_end_list = []
    with open(config_reading.in_file_path, "r") as f_in:
        num_lines = count_texts_in_file(config_reading, f_in)
    segment_start_end_list = create_segment_start_end_list_of_quantity(num_lines, num_segments)
    return segment_start_end_list


def get_percentage_segment_dict(i_start, i_end):
    percentage_segmens = create_segment_start_end_list_of_quantity(i_end - i_start, 100)
    percentage_segmens = [e + i_start - 1 for s,e in percentage_segmens]
    percentage_segment_dict = {}
    for percentage, i_segment in enumerate(percentage_segmens, start=1):
        percentage_segment_dict[i_segment] = percentage
    return percentage_segment_dict


def adapt_config_to_tmp(config_writing, config_processing, process_id):
    if config_processing.cpu_count > 1:
        config_writing_per_process = copy.copy(config_writing)
        if type(config_writing_per_process) is ConfigWritingClean:
            config_writing_per_process.config_writing_clean = copy.copy(config_writing.config_writing_clean)
            config_writing_per_process.config_writing_dirty = copy.copy(config_writing.config_writing_dirty)
            config_writing_per_process.config_writing_clean.out_file_path = (
                TMP_FOLDER 
                + "tmp_clean_"
                + str(process_id) 
                + "." 
                + get_filetype_of_config(config_writing.config_writing_clean)
            )
            config_writing_per_process.config_writing_dirty.out_file_path = (
                TMP_FOLDER 
                + "tmp_dirty_" 
                + str(process_id) 
                + "." 
                + get_filetype_of_config(config_writing.config_writing_dirty)
            )
        else:
            config_writing_per_process.out_file_path = (
                TMP_FOLDER 
                + "tmp_" 
                + str(process_id)
                + "." 
                + get_filetype_of_config(config_writing)
            )
    else:
        config_writing_per_process = config_writing
    return config_writing_per_process


def adapt_config_to_file_type(config_reading_or_writing):
    if type(config_reading_or_writing) is ConfigWritingClean:
        config_reading_or_writing.config_writing_clean = adapt_config_to_file_type(config_reading_or_writing.config_writing_clean)
        config_reading_or_writing.config_writing_dirty = adapt_config_to_file_type(config_reading_or_writing.config_writing_dirty)
    else:
        try:
            file_path = config_reading_or_writing.in_file_path
        except:
            pass
        try:
            file_path = config_reading_or_writing.out_file_path
        except:
            pass
        file_type = file_path.split(".")[-1]
        if file_type == "txt":
            if type(config_reading_or_writing) is ConfigReading:
                config_reading_or_writing = ConfigReadingTxt(**asdict(config_reading_or_writing))
            elif type(config_reading_or_writing) is ConfigWriting:
                config_reading_or_writing = ConfigWritingTxt(**asdict(config_reading_or_writing))
    return config_reading_or_writing


def processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
    func_reading = get_func_reading(config_reading)
    func_processing = get_func_processing(config_processing)
    percentage_segment_dict = get_percentage_segment_dict(start_end_segment[0], start_end_segment[1])
    for i_text, text in func_reading(config_reading, f_in, start_end_segment):
        text_processed = func_processing(config_processing, text)
        if percentage := percentage_segment_dict.get(i_text):
            print(f"process_id: {process_id}: {percentage}%")
        yield text_processed


def processing_context_common(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with open(config_reading.in_file_path, "r") as f_in, open(config_writing.out_file_path) as f_out:
        func_writing = get_func_writing(config_writing)
        for text_processed in processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
            func_writing(config_writing, text_processed, f_out)


def processing_context_clean(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with (
        open(config_reading.in_file_path, "r") as f_in, 
        open(config_writing.config_writing_clean.out_file_path, "w") as f_out_clean,
        open(config_writing.config_writing_dirty.out_file_path, "w") as f_out_dirty,
    ):
        func_writing_clean = get_func_writing(config_writing.config_writing_clean)
        func_writing_dirty = get_func_writing(config_writing.config_writing_dirty)
        for text_processed, is_text_processed_clean in processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
            if is_text_processed_clean:
                func_writing_clean(config_writing.config_writing_clean, text_processed, f_out_clean)
            else:
                func_writing_dirty(config_writing.config_writing_dirty, text_processed, f_out_dirty)


def main_process_multi(config_processing, config_reading, config_writing):
    DEBUG_SINGLE_PROCESS = True
    print("- all processing start ----------------------------------------------")
    segment_start_end_list = create_segment_start_end_list_of_file(
        config_reading, 
        config_processing.cpu_count
    )

    if type(config_processing) in [ConfigProcessingChangeCase]:
        func_processing = processing_context_common
    elif type(config_processing) is ConfigProcessingClean:
        func_processing = processing_context_clean

    process_list = []
    for process_id, segment_start_end in enumerate(segment_start_end_list):
        config_writing_per_process = adapt_config_to_tmp(config_writing, config_processing, process_id)
        if DEBUG_SINGLE_PROCESS:
            func_processing(
                config_processing,
                config_reading,
                config_writing_per_process,
                process_id, 
                segment_start_end, 
            )
        else:
            process = Process(
                target=func_processing,
                args=(
                    process_id, 
                    config_processing,
                    config_reading,
                    config_writing_per_process,
                    segment_start_end, 
                )
            )
            process.start()
            process_list.append(process)
    if not DEBUG_SINGLE_PROCESS:
        for process in process_list:
            process.join()
    print("- all processing done -----------------------------------------------")
    if config_processing.cpu_count > 1:
        merge_tmp_main(config_writing)


def main():

    # config reading
    print("- preparing --------------------------------------------------------")
    config_processing = get_config_processing()
    config_reading = get_config_reading()
    config_writing = get_config_writing()
    config_writing_metadata = get_config_writing_metadata()

    # config adaptions and calling into main_process_multi
    if config_reading.in_file_path:
        config_reading = adapt_config_to_file_type(config_reading)
        config_writing = adapt_config_to_file_type(config_writing)
        main_process_multi(config_processing, config_reading, config_writing)
    else:
        for file in os.listdir(config_reading.in_folder):
            config_reading.in_file_path = config_reading.in_folder + file
            if type(config_writing) is ConfigWritingClean:
                file_split = file.split(".")
                file_name = "".join(file_split[:-1])
                file_type = file_split[-1]
                config_writing.config_writing_clean.out_file_path = (
                    config_writing.out_folder
                    + file_name
                    + "_clean."
                    + file_type
                )
                config_writing.config_writing_dirty.out_file_path = (
                    config_writing.out_folder
                    + file_name
                    + "_dirty."
                    + file_type
                )
            else:
                config_writing.out_file_path = config_writing.out_folder + file
            config_reading = adapt_config_to_file_type(config_reading)
            config_writing = adapt_config_to_file_type(config_writing)
            main_process_multi(config_processing, config_reading, config_writing)

    # write metadata
    if config_writing_metadata.out_metadata_file_path:
        write_veld_data_yaml(config_writing_metadata, config_writing)


if __name__ == "__main__":
    main()

