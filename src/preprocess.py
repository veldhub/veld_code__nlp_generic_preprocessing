import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from multiprocessing import Process
from time import sleep

import yaml


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/data/"
OUT_METADATA_FOLDER = "/veld/output/metadata/"


@dataclass
class ConfigReading:
    in_folder: str = None
    in_file_path: str = None


@dataclass
class ConfigReadingTxt(ConfigReading):
    pass


@dataclass
class ConfigProcessing:
    cpu_count: int = None
    sleep_duration: int = None


@dataclass
class ConfigProcessingChangeCase(ConfigProcessing):
    set_case: str = None


@dataclass
class ConfigProcessingClean(ConfigProcessing):
    min_char_percentage: int | float = None


@dataclass
class ConfigWriting:
    out_folder: str = None
    out_file_path: str = None
    out_metadata_folder: str = None
    out_metadata_file_path: str = None
    out_metadata_description: str = None
    out_metadata_topic: str | list[str] = None
    out_metadata_content: str | list[str] = None


@dataclass
class ConfigWritingTxt(ConfigWriting):
    pass


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
    out_metadata_file = get_env_var("out_metadata_file")
    if out_metadata_file:
        out_metadata_file_path = OUT_METADATA_FOLDER + out_metadata_file
    else:
        out_metadata_file_path = None
    config_writing = ConfigWriting(
        out_folder=OUT_FOLDER,
        out_file_path=out_file_path,
        out_metadata_folder=OUT_METADATA_FOLDER,
        out_metadata_file_path=out_metadata_file_path,
        out_metadata_description=get_env_var("out_metadata_description"),
        out_metadata_topic=get_env_var_to_list("out_metadata_topic"),
        out_metadata_content=get_env_var_to_list("out_metadata_content"),
    )
    return config_writing


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
    elif processing_func_name == "clean":
        config_processing = ConfigProcessingClean(
            **asdict(config_processing),
            min_char_percentage=get_env_var("min_char_percentage"),
        )
    return config_processing


def get_func_reading(config_reading):
    return func_reading


def func_reading_txt(f_in, i_start, i_end):
    for i_line, line in enumerate(f_in):
        if i_line >= i_start:
            if i_line < i_end:
                yield (i_line, line)
            else:
                break


def func_shared_processing_individual(func_reading, func_processing, config_reading, config_processing, process_id, in_file):
    for i_text, text in func_reading(config_reading):
        percentage_current = percentage_segment_dict.get(i_text)
        if percentage_current:
            print(f"process_id: {process_id}: at {percentage_current}%")
        text_processed = func_processing(text, config_processing)
        yield text_processed


def func_shared_processing_simple(config_processing, config_reading, config_writing, process_id):
    func_processing = get_func_processing(config_processing)
    func_reading = get_func_reading(config_reading)
    func_writing = get_func_writing(config_writing)
    with open(config_reading.in_file_path, "r") as in_file, open(out_file_path, "w") as out_file:
        for text_processed in func_shared_processing_individual(func_reading, func_processing, config_reading, config_processing, in_file):
            func_writing(text_processed, config_writing, out_file)


def func_processing_change_case(text, config_processing):
    if config_processing.set_case == "upper":
        func_case = str.upper
    elif config_processing.set_case == "lower":
        func_case = str.lower
    text_processed = func_case(text)
    return text_processed


def func_processing_clean_individual(text, config_processing):

    def func_processing_clean_clean(text):
        return process_text(text, True)

    def func_processing_clean_dirty(text):
        return process_text(text, False)

    count_clean = 0
    count_dirty = 0
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("L") or cat.startswith("Z"):
            count_clean += 1
        else:
            count_dirty += 1
    percentage_char = (100 * count_clean)  / (count_clean + count_dirty)
    is_text_clean = percentage_char >= MIN_PERCENTAGE_CHAR
    if (is_text_clean and set_write_clean) or (not is_text_clean and not set_write_clean):
        text_to_write = text
    elif (is_text_clean and not set_write_clean) or (not is_text_clean and set_write_clean):
        text_to_write = ""
    return text_to_write


def func_processing_clean(text, config_processing, config_reading, config_writing, process_id):
    config_writing_clean = config_writing.config_writing_clean
    config_writing_dirty = config_writing.config_writing_dirty
    func_reading = get_func_reading(config_reading)
    func_writing_clean = get_func_writing(config_writing_clean)
    func_writing_dirty = get_func_writing(config_writing_dirty)
    with (
        open(config_reading.in_file_path, "r") as in_file, 
        open(config_writing_clean.out_file_path, "w") as out_file_clean, 
        open(config_writing_dirty.out_file_path, "w") as out_file_dirty,
    ):
        for text_processed in func_shared_processing_individual(func_reading, func_processing, config_reading, config_processing):
            func_writing_clean(text_processed_clean, config_writing_clean)
            func_writing_dirty(text_processed_dirty, config_writing_dirty)


def func_writing_txt(text, f_out):
    f_out.write(text)


def count_lines(file_path):
    result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
    num_lines = int(result.stdout.split()[0])
    return num_lines


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


def create_segment_start_end_list_of_in_file(config_reading, num_segments):
    segment_start_end_list = []
    if type(config_reading) is ConfigReadingTxt:
        print("counting lines of file.")
        num_lines = count_lines(config_reading.in_file_path)
        if num_lines == 0:
            num_lines = 1
        print(f"input file has {num_lines} lines")
        segment_start_end_list = create_segment_start_end_list_of_quantity(num_lines, num_segments)
    return segment_start_end_list


def merge_tmp(tmp_folder, out_file_path):
    print("joining tmp files into one.")
    with open(out_file_path, "w") as f_out:
        tmp_file_path_list = sorted([tmp_folder + "/" + f for f in os.listdir(tmp_folder)])
        for tmp_file_path in tmp_file_path_list:
            with open(tmp_file_path, "r") as f_in:
                for line in f_in:
                    f_out.write(line)


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



def get_percentage_segment_dict(i_start, i_end):
    print_segments = create_segment_start_end_list_of_quantity(i_end - i_start, 100)
    print_segments = [e + i_start - 1 for s,e in print_segments]
    print_segments_dict = {}
    for percentage, i_segment in enumerate(print_segments, start=1):
        print_segments_dict[i_segment] = percentage


def main_process_single(
    process_id, 
    config_processing,
    config_reading,
    config_writing,
    # tmp_folder,
    # segment_start_end, 
):
    print(f"- process_id: {process_id}: start ----------------------------------------------")

    if type(config_processing) is ConfigProcessingChangeCase:
        func_shared_processing_simple(config_processing, config_reading, config_writing)
    elif type(config_processing) is ConfigProcessingClean:
        func_processing_clean(config_processing, config_reading, config_writing)

    # load matching functions
    # if type(config_reading) is ConfigReadingTxt:
    #     func_reading = func_reading_txt
    # if type(config_processing) is ConfigProcessingChangeCase:
    #     func_processing = func_processing_change_case
    # elif type(config_processing) is ConfigProcessingClean:
    #     func_processing = func_processing_clean
    # if type(config_writing) is ConfigWritingTxt:
    #     func_writing = func_writing_txt
    # 
    # # main single core processing
    # i_start = segment_start_end[0]
    # i_end = segment_start_end[1]
    # percentage_segment_dict = get_percentage_segment_dict(i_start, i_end)
    # if config_processing.cpu_count > 1:
    #     if type(config_writing) is ConfigWritingTxt:
    #         out_file_path = tmp_folder + str(process_id) + ".txt"
    # else:
    #     out_file_path = config_writing.out_file_path
    # with open(config_reading.in_file_path, "r") as in_file, open(out_file_path, "w") as out_file:
    #     for i_text, text in func_reading(in_file, i_start, i_end):
    #         percentage_current = percentage_segment_dict.get(i_text)
    #         if percentage_current:
    #             print(f"process_id: {process_id}: at {percentage_current}%")
    #         func_processing(text, config_processing, func_writing, out_file)
    print(f"- process_id: {process_id}: done -----------------------------------------------")


def main_process_multi(config_processing, config_reading, config_writing):
    DEBUG_SINGLE_PROCESS = True
    print("- all processing start ----------------------------------------------")
    segment_start_end_list = create_segment_start_end_list_of_in_file(
        config_reading, 
        config_processing.cpu_count
    )
    process_list = []
    tmp_folder = "/tmp/"
    for process_id, segment_start_end in enumerate(segment_start_end_list):
        if DEBUG_SINGLE_PROCESS:
            main_process_single(
                process_id, 
                config_processing,
                config_reading,
                config_writing,
                tmp_folder,
                segment_start_end, 
            )
        else:
            process = Process(
                target=main_process_single,
                args=(
                    process_id, 
                    config_processing,
                    config_reading,
                    config_writing,
                    tmp_folder,
                    segment_start_end, 
                )
            )
            process.start()
            process_list.append(process)
            sleep(config_processing.sleep_duration)
    if not DEBUG_SINGLE_PROCESS:
        for process in process_list:
            process.join()
    print("- all processing done -----------------------------------------------")
    if config_processing.cpu_count > 1:
        merge_tmp(tmp_folder, config_writing.out_file_path)


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

