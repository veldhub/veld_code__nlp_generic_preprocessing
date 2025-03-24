import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from multiprocessing import Process
from time import sleep


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/data/"
OUT_METADATA_FOLDER = "/veld/output/metadata/"


@dataclass
class ConfigReading:
    in_folder: str = None
    in_file_path: str = None
    # in_file_type: str = None


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
    threshold: float = None


@dataclass
class ConfigWriting:
    out_folder: str = None
    out_file_path: str = None
    # out_file_type: str = None
    out_data_description: str = None


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
        out_data_description=get_env_var("out_data_description"),
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
    return config_processing


def func_reading_txt(f_in, i_start, i_end):
    for i_line, line in enumerate(f_in):
        if i_line >= i_start:
            if i_line < i_end:
                yield line
            else:
                break


def func_processing_change_case(text, config_processing):
    if config_processing.set_case == "upper":
        func_case = str.upper
    elif config_processing.set_case == "lower":
        func_case = str.lower
    text_processed = func_case(text)
    return text_processed


def func_processing_clean(text, config_processing):
    pass


def func_writing_txt(text, f_out):
    f_out.write(text)


def create_segment_start_end_list(config_reading, num_segments):
    segment_start_end_list = []
    if type(config_reading) is ConfigReadingTxt:

        print("counting lines of file.")
        result = subprocess.run(["wc", "-l", config_reading.in_file_path], capture_output=True, text=True)
        num_lines = int(result.stdout.split()[0]) + 1
        print(f"input file has {num_lines} lines")

        step = num_lines // num_segments
        i_start = 0
        for i_segment in range(1, num_segments + 1):
            if i_segment < num_segments:
                i_end = i_segment * step
                segment_start_end_list.append((i_start, i_end))
                i_start = i_end
            else:
                segment_start_end_list.append((i_start, num_lines))

    return segment_start_end_list


def merge_tmp(tmp_folder, out_file_path):
    print("joining tmp files into one.")
    with open(out_file_path, "w") as f_out:
        for tmp_file_path in [tmp_folder + "/" + f for f in os.listdir(tmp_folder)]:
            with open(tmp_file_path, "r") as f_in:
                for line in f_in:
                    f_out.write(line)


def main_process_single(
    process_id, 
    config_processing,
    config_reading,
    config_writing,
    tmp_folder,
    segment_start_end, 
):
    i_start = segment_start_end[0]
    i_end = segment_start_end[1]
    
    if type(config_reading) is ConfigReadingTxt:
        func_reading = func_reading_txt

    if type(config_processing) is ConfigProcessingChangeCase:
        func_processing = func_processing_change_case

    if type(config_writing) is ConfigWritingTxt:
        func_writing = func_writing_txt

    with open(config_reading.in_file_path, "r") as in_file:
        print("- processing ---------------------------------------------")
        if config_processing.cpu_count > 1:
            print(f"process_id: {process_id}")
            if type(config_writing) is ConfigWritingTxt:
                out_file_path = tmp_folder + str(process_id) + ".txt"
        else:
            out_file_path = config_writing.out_file_path
        with open(out_file_path, "w") as out_file:
            for text in func_reading(in_file, i_start, i_end):
                text_processed = func_processing(text, config_processing)
                func_writing(text_processed, out_file)


def main_process_multi(config_processing, config_reading, config_writing):
    file_type = config_reading.in_file_path.split(".")[-1]
    if file_type == "txt" and type(config_reading) is not ConfigReadingTxt:
        config_reading = ConfigReadingTxt(**asdict(config_reading))
    if file_type == "txt" and type(config_writing) is not ConfigWritingTxt:
        config_writing = ConfigWritingTxt(**asdict(config_writing))

    segment_start_end_list = create_segment_start_end_list(
        config_reading, 
        config_processing.cpu_count
    )
    process_list = []
    tmp_folder = "/tmp/"

    for process_id, segment_start_end in enumerate(segment_start_end_list):
        # main_process_single(
        #     process_id, 
        #     config_processing,
        #     config_reading,
        #     config_writing,
        #     tmp_folder,
        #     segment_start_end, 
        # )
        # sleep(config_processing.sleep_duration)
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
    for process in process_list:
        process.join()
    print("- processing done --------------------------------------------------")
    if config_processing.cpu_count > 1:
        merge_tmp(tmp_folder, config_writing.out_file_path)


def main():

    print("- reading configuration ---------------------------------------------")
    config_processing = get_config_processing()
    config_reading = get_config_reading()
    config_writing = get_config_writing()

    if config_reading.in_file_path and config_writing.out_file_path:
        main_process_multi(config_processing, config_reading, config_writing)
    elif not config_reading.in_file_path and not config_writing.out_file_path:
        for file in os.listdir(config_reading.in_folder):
            config_reading.in_file_path = config_reading.in_folder + file
            config_writing.out_file_path = config_writing.out_folder + file
            main_process_multi(config_processing, config_reading, config_writing)
    else:
        # TODO
        raise Exception("")


if __name__ == "__main__":
    main()
