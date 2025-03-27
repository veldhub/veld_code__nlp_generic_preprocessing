import copy
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from multiprocessing import Process
from time import sleep

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
class ConfigWritingMetadata():
    out_metadata_folder: str = None
    out_metadata_file_path: str = None
    out_metadata_description: str = None
    out_metadata_topic: str | list[str] = None
    out_metadata_content: str | list[str] = None


@dataclass
class ConfigProcessing:
    cpu_count: int = None
    sleep_duration: int = None


@dataclass
class ConfigProcessingChangeCase(ConfigProcessing):
    set_case: str = None


@dataclass
class ConfigProcessingRegexReplace(ConfigProcessing):
    regex_sub: Tuple[str, str] = None


@dataclass
class ConfigProcessingClean(ConfigProcessing):
    min_clean_char_percentage: int | float = None


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


def get_config_reading():
    in_file = get_env_var("in_file")
    if in_file:
        in_file_path = IN_FOLDER + in_file
    else:
        in_file_path = None
    config_reading = ConfigReading(
        in_folder=IN_FOLDER,
        in_file_path=in_file_path,
    )
    return config_reading


def get_config_writing():
    out_file = get_env_var("out_file")
    if out_file:
        out_file_path = OUT_FOLDER + out_file
    else:
        out_file_path = None
    config_writing = ConfigWriting(
        out_folder=OUT_FOLDER,
        out_file_path=out_file_path,
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
    processing_func_name = sys.argv[1]
    print(f"processing: {processing_func_name}")
    config_processing = ConfigProcessing(
        cpu_count=get_env_var("cpu_count", int),
        sleep_duration=get_env_var("sleep_duration", int),
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
            min_clean_char_percentage=get_env_var("min_clean_char_percentage"),
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


def write_veld_data_yaml(config_writing):
    result = subprocess.run(["du", "-sh", config_writing.out_folder], capture_output=True, text=True)
    data_size = result.stdout.split()[0]
    if config_writing.out_file_path:
        num_lines = count_lines(config_writing.out_file_path)
    else:
        num_lines = 0
        for file in os.listdir(config_writing.out_folder):
            num_lines += count_lines(config_writing.out_folder + file)
    if type(config_writing) is ConfigWritingTxt:
        file_type = "txt"
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


def merge_tmp(out_file_path):
    print("joining tmp files into one.")
    with open(out_file_path, "w") as f_out:
        tmp_file_path_list = sorted([TMP_FOLDER + "/" + f for f in os.listdir(TMP_FOLDER)])
        for tmp_file_path in tmp_file_path_list:
            with open(tmp_file_path, "r") as f_in:
                for line in f_in:
                    f_out.write(line)


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


def create_segment_start_end_list_of_file(config_reading, num_segments):
    print("- creating index segments of file -----------------------------------")
    segment_start_end_list = []
    with open(config_reading.in_file_path, "r") as f_in:
        func_reading = get_func_reading(config_reading)
        i_line = 0
        for i_line, _ in func_reading(config_reading):
            pass
    num_lines = i_line + 1
    segment_start_end_list = create_segment_start_end_list_of_quantity(num_lines, num_segments)
    return segment_start_end_list


def get_percentage_segment_dict(i_start, i_end):
    percentage_segmens = create_segment_start_end_list_of_quantity(i_end - i_start, 100)
    percentage_segmens = [e + i_start - 1 for s,e in percentage_segmens]
    percentage_segment_dict = {}
    for percentage, i_segment in enumerate(percentage_segmens, start=1):
        percentage_segment_dict[i_segment] = percentage
    return percentage_segment_dict


def processing_execution(config_processing, config_reading, start_end_segment):
    func_reading = get_func_reading(config_reading)
    func_processing = get_func_processing(config_processing)
    percentage_segment_dict = get_percentage_segment_dict(start_end_segment[0], start_end_segment[1])
    for i_text, text in func_reading(config_reading, f_in, start_end_segment):
        text_processed = func_processing(config_processing, text)
        if percentage := percentage_segment_dict.get(i)
            print(f"process_id: {process_id}: {percentage}%")
        yield text_processed


def processing_context_common(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with open(config_reading.in_file_path, "r") as f_in, open(config_writing.out_file_path) as f_out:
        func_writing = get_func_writing(config_writing)
        for text in processing_execution(config_processing, config_reading, start_end_segment):
            func_writing(config_writing, text_processed, f_out)


def processing_context_clean(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with (
        open(config_reading.in_file_path, "r") as f_in, 
        open(config_writing.config_writing_clean.out_file_path) as f_out_clean
        open(config_writing.config_writing_dirty.out_file_path) as f_out_dirty
    ):
        func_writing = get_func_writing(config_writing)
        for text_processed, is_text_processed_clean in processing_execution(config_processing, config_reading, start_end_segment):
            if is_text_processed_clean:
                func_writing(config_writing.config_writing_clean, text_processed_clean, f_out_clean)
            else:
                func_writing(config_writing.config_writing_dirty, text_processed_dirty, f_out_dirty)


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
        if config_processing.cpu_count > 1:
            config_writing_per_process = copy.copy(config_writing)
            config_writing_per_process.out_file_path = TMP_FOLDER + "tmp_" + str(process_id) + get_filetype_of_config(config_writing)
        else:
            config_writing_per_process = config_writing
        if DEBUG_SINGLE_PROCESS:
            func_processing(
                process_id, 
                config_processing,
                config_reading,
                config_writing_per_process,
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
        merge_tmp(config_writing.out_file_path)


def adapt_config_to_file_type(config_reading, config_writing):
    in_file_type = config_reading.in_file_path.split(".")[-1]
    if in_file_type == "txt" and type(config_reading) is not ConfigReadingTxt:
        config_reading = ConfigReadingTxt(**asdict(config_reading))
    out_file_type = config_writing.out_file_path.split(".")[-1]
    if out_file_type == "txt" and type(config_writing) is not ConfigWritingTxt:
        config_writing = ConfigWritingTxt(**asdict(config_writing))
    return config_reading, config_writing


def main():

    # config reading
    print("- preparing --------------------------------------------------------")
    config_processing = get_config_processing()
    config_reading = get_config_reading()
    config_writing = get_config_writing()
    cofnig_writing_metadata = get_config_writing_metadata()

    # config adaptions and calling into main_process_multi
    if config_reading.in_file_path and config_writing.out_file_path:
        config_reading, config_writing = adapt_config_to_file_type(config_reading, config_writing)
        main_process_multi(config_processing, config_reading, config_writing)
    elif not config_reading.in_file_path and not config_writing.out_file_path:
        for file in os.listdir(config_reading.in_folder):
            config_reading.in_file_path = config_reading.in_folder + file
            config_writing.out_file_path = config_writing.out_folder + file
            config_reading, config_writing = adapt_config_to_file_type(
                config_reading, 
                config_writing
            )
            main_process_multi(config_processing, config_reading, config_writing)
        config_reading.in_file_path = None
        config_writing.out_file_path = None

    # write metadata
    if config_writing.out_metadata_file_path:
        write_veld_data_yaml(config_writing)


if __name__ == "__main__":
    main()

