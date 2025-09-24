import copy
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, asdict
from multiprocessing import Process
from typing import List

import spacy
import yaml
from nltk.tokenize import sent_tokenize


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/data/"
OUT_METADATA_FOLDER = "/veld/output/metadata/"
TMP_FOLDER = "/veld/output/tmp/"


@dataclass
class ConfigReading:
    folder: str = None
    file_path: str = None
    ignore_file: List[str] = None


@dataclass
class ConfigReadingTxt(ConfigReading):
    txt_has_lines: bool = None


@dataclass
class ConfigWriting:
    folder: str = None
    file_path: str = None


@dataclass
class ConfigWritingTxt(ConfigWriting):
    set_delimit_by_newline: bool = False


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
    regex_pattern_match: str = None
    regex_pattern_replacement: str = None


@dataclass
class ConfigProcessingClean(ConfigProcessing):
    min_clean_char_percentage: float = None
    append_to_file: bool = True


@dataclass
class ConfigProcessingSplitSentences(ConfigProcessing):
    language: str = None


@dataclass
class ConfigProcessingLemmatize(ConfigProcessing):
    nlp: spacy.lang = None


def get_env_var(var_name, cast_func=None, mandatory=False, default=None):
    var_content = os.getenv(var_name)
    if var_content is not None:
        print(f"{var_name}: {var_content.__repr__()}")
    elif default is not None:
        var_content = default
    elif mandatory:
        raise Exception(f"environment variable: '{var_name}' is mandatory")
    if cast_func:
        try:
            var_content = cast_func(var_content)
        except:
            raise Exception(f"Could not convert var '{var_name}' to {cast_func}")
    return var_content


def get_env_var_to_list(var_name):
    var_content = get_env_var(var_name)
    if var_content:
        var_content_list = var_content.split(",")
        return var_content_list
    else:
        return None


def get_env_var_to_bool(var_name):
    var_content = get_env_var(var_name)
    if var_content:
        if var_content.lower() == "true":
            return True
        elif var_content.lower() == "false":
            return False
        else:
            raise Exception(f"Wrong value for a boolean variable: {var_content}")
    else:
        return None


def concatenate_folder_and_file(folder, file):
    if file:
        return folder + file
    else:
        return None


def get_processing_func_name():
    processing_func_name = sys.argv[1]
    return processing_func_name


def create_config_reading():
    config_reading = ConfigReading(
        folder=IN_FOLDER,
        file_path=concatenate_folder_and_file(IN_FOLDER, get_env_var("in_file")),
        ignore_file=get_env_var_to_list("ignore_file"),
    )
    txt_has_lines = get_env_var_to_bool("txt_has_lines")
    if txt_has_lines is not None:
        config_reading = ConfigReadingTxt(
            **asdict(config_reading),
            txt_has_lines=txt_has_lines,
        )
    return config_reading


def create_config_writing():
    processing_func_name = get_processing_func_name()
    if processing_func_name == "clean":
        config_writing = ConfigWritingClean(
            config_writing_clean=ConfigWriting(
                folder="/veld/output/data_clean/",
                file_path=concatenate_folder_and_file("/veld/output/data_clean/", get_env_var("out_file_clean")),
            ),
            config_writing_dirty=ConfigWriting(
                folder="/veld/output/data_dirty/",
                file_path=concatenate_folder_and_file("/veld/output/data_dirty/", get_env_var("out_file_dirty")),
            ),
        )
        if not config_writing.config_writing_dirty.file_path:
            config_writing.config_writing_dirty.file_path = "/tmp/tmp_dirty" 
    elif processing_func_name == "split_sentences":
        config_writing = ConfigWritingTxt(
            folder=OUT_FOLDER,
            file_path=concatenate_folder_and_file(OUT_FOLDER, get_env_var("out_file")),
            set_delimit_by_newline=True,
        )
    else:
        config_writing = ConfigWriting(
            folder=OUT_FOLDER,
            file_path=concatenate_folder_and_file(OUT_FOLDER, get_env_var("out_file")),
        )
    return config_writing


def create_config_writing_metadata():
    out_metadata_file = get_env_var("out_metadata_file")
    if out_metadata_file:
        out_metadata_file_path = OUT_METADATA_FOLDER + out_metadata_file
        config_writing_metadata = ConfigWritingMetadata(
            out_metadata_folder=OUT_METADATA_FOLDER,
            out_metadata_file_path=out_metadata_file_path,
            out_metadata_description=get_env_var("out_metadata_description"),
            out_metadata_topic=get_env_var_to_list("out_metadata_topic"),
            out_metadata_content=get_env_var_to_list("out_metadata_content"),
        )
        return config_writing_metadata
    else:
        return None


def create_config_processing():
    processing_func_name = get_processing_func_name()
    print(f"processing: {processing_func_name}")
    max_cpu_count = os.cpu_count()
    if max_cpu_count >= 4:
        max_cpu_count -= 2
    else:
        max_cpu_count = 1
    config_processing = ConfigProcessing(
        cpu_count=get_env_var("cpu_count", int, default=max_cpu_count),
    )
    if processing_func_name == "change_case":
        config_processing = ConfigProcessingChangeCase(
            **asdict(config_processing),
            set_case=get_env_var("set_case"),
        )
    elif processing_func_name == "remove_punctuation":
        config_processing = ConfigProcessingRegexReplace(
            **asdict(config_processing),
            regex_pattern_match=r"[^\w\s]",
            regex_pattern_replacement="",
        )
    elif processing_func_name == "regex_replace": 
        regex_pattern_match = get_env_var("regex_pattern_match", mandatory=True)
        regex_pattern_replacement = get_env_var("regex_pattern_replacement", mandatory=True)
        config_processing = ConfigProcessingRegexReplace(
            **asdict(config_processing),
            # regex_sub=(regex_pattern_match, regex_pattern_replacement),
            regex_pattern_match=regex_pattern_match,
            regex_pattern_replacement=regex_pattern_replacement,
        )
    elif processing_func_name == "clean":
        config_processing = ConfigProcessingClean(
            **asdict(config_processing),
            min_clean_char_percentage=get_env_var("min_percentage_char", float),
            append_to_file=get_env_var_to_bool("append_to_file"),
        )
    elif processing_func_name == "split_sentences":
        config_processing = ConfigProcessingSplitSentences(
            **asdict(config_processing),
            language=get_env_var("language", mandatory=True),
        )
    elif processing_func_name == "lemmatize":
        nlp = spacy.load(get_env_var("spacy_model", mandatory=True))
        config_processing = ConfigProcessingLemmatize(
            **asdict(config_processing),
            nlp=nlp,
        )
    else:
        raise Exception(
            f"could not determine config_processing for func_name: {processing_func_name}"
        )
    return config_processing


def get_func_reading(config_reading):
    if type(config_reading) in [ConfigReadingTxt, ConfigWritingTxt]:
        return func_reading_txt
    else:
        raise Exception(f"no registered function for {config_reading}")


def get_func_writing(config_writing):
    if type(config_writing) is ConfigWritingTxt:
        return func_writing_txt
    else:
        raise Exception(f"no registered function for {config_writing}")


def get_func_processing(config_processing):
    if type(config_processing) is ConfigProcessingChangeCase:
        return func_processing_change_case
    elif type(config_processing) is ConfigProcessingClean:
        return func_processing_clean
    elif type(config_processing) is ConfigProcessingRegexReplace:
        return func_processing_regex_replace
    elif type(config_processing) is ConfigProcessingSplitSentences:
        return func_processing_split_sentences
    elif type(config_processing) is ConfigProcessingLemmatize:
        return func_processing_lemmatize
    else:
        raise Exception(f"no registered function for {config_processing}")


def get_func_context_processing(config_processing):
    if type(config_processing) in [
        ConfigProcessingChangeCase,
        ConfigProcessingRegexReplace,
        ConfigProcessingSplitSentences,
        ConfigProcessingLemmatize,
    ]:
        return processing_context_common
    elif type(config_processing) is ConfigProcessingClean:
        return processing_context_clean
    else:
        raise Exception(f"no registered context function for: {config_processing}")


def get_filetype_of_config(config_reading_or_writing):
    if type(config_reading_or_writing) in [ConfigReadingTxt, ConfigWritingTxt]:
        return "txt"


def func_reading_txt(config_reading, f_in, segment_start_end_list=None):
    for i_text, text in enumerate(f_in):
        if segment_start_end_list:
            if i_text >= segment_start_end_list[0]:
                if i_text < segment_start_end_list[1]:
                    yield (i_text, text)
                else:
                    break
        else:
            yield (i_text, text)
        i_text += 1


def func_writing_txt(config_writing, text, f_out):
    if config_writing.set_delimit_by_newline:
        text += "\n"
    f_out.write(text)


def func_processing_change_case(config_processing, text):
    if config_processing.set_case == "upper":
        func_case = str.upper
    elif config_processing.set_case == "lower":
        func_case = str.lower
    yield func_case(text)


def func_processing_clean(config_processing, text):
    count_clean = 0
    count_dirty = 0
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("Z"):
            count_clean += 1
        else:
            count_dirty += 1
    percentage_clean_char = (100 * count_clean)  / (count_clean + count_dirty)
    if percentage_clean_char >= config_processing.min_clean_char_percentage:
        yield (text, True)
    else:
        yield (text, False)


def func_processing_regex_replace(config_processing, text):
    yield re.sub(
        config_processing.regex_pattern_match,
        config_processing.regex_pattern_replacement,
        text,
    )


def func_processing_split_sentences(config_processing, text):
    text_processed_list = sent_tokenize(text, language=config_processing.language)
    for text_processed in text_processed_list:
        yield text_processed


def func_processing_lemmatize(config_processing, text):
    doc = config_processing.nlp(text)
    yield " ".join([t.lemma_ for t in doc])


def write_veld_data_yaml(config_writing_metadata, config_writing):
    if type(config_writing) is ConfigWritingClean:
        folder_or_file = config_writing.config_writing_clean.file_path
        if not folder_or_file:
            folder_or_file = config_writing.config_writing_clean.folder
    else:
        folder_or_file = config_writing.folder
        if not folder_or_file:
            config_writing.folder
    result = subprocess.run(["du", "-sh", folder_or_file], capture_output=True, text=True)
    data_size = result.stdout.split()[0]
    num_lines = count_texts_of_output_main(config_writing)
    veld_data_yaml = {
        "x-veld": {
            "data": {
                "description": config_writing_metadata.out_metadata_description,
                "topic": config_writing_metadata.out_metadata_topic,
                "content": config_writing_metadata.out_metadata_content,
                "file_type": get_filetype_of_config(config_writing),
                "additional": {
                    "data size": data_size,
                    "number of lines": num_lines,
                }
            }
        }
    }
    with open(config_writing_metadata.out_metadata_file_path, "w") as f:
        yaml.dump(veld_data_yaml, f, sort_keys=False)


def merge_tmp_individual(config_writing, file_name_pattern=None):
    with open(config_writing.file_path, "w") as f_out:
        tmp_file_path_list = []
        for file in os.listdir(TMP_FOLDER):
            file_name = file.split(".")[0]
            number = ""
            for char in file_name[::-1]:
                try:
                    int(char)
                except:
                    break
                else:
                    number += char
            number = int(number[::-1])
            tmp_file_path_list.append((number, TMP_FOLDER + file))
        tmp_file_path_list = sorted(tmp_file_path_list, key=lambda x: x[0])
        for _, tmp_file_path in tmp_file_path_list:
            if not file_name_pattern or file_name_pattern in tmp_file_path:
                with open(tmp_file_path, "r") as f_in:
                    f_out.write(f_in.read())


def merge_tmp_main(config_writing):
    print("joining tmp files into one.")
    if type(config_writing) is ConfigWritingClean:
        merge_tmp_individual(config_writing.config_writing_clean, "clean") 
        merge_tmp_individual(config_writing.config_writing_dirty, "dirty") 
    else:
        merge_tmp_individual(config_writing)


def count_texts_of_output_individual(config_writing, file_name_pattern=None):
    num_lines = 0
    if config_writing.file_path:
        num_lines = count_texts_in_file(config_writing)
    else:
        num_lines = 0
        for file in os.listdir(config_writing.folder):
            config_writing.file_path = config_writing.folder + file
            if not file_name_pattern or file_name_pattern in file:
                num_lines += count_texts_in_file(config_writing)
    return num_lines


def count_texts_of_output_main(config_writing):
    num_lines = 0
    if type(config_writing) is ConfigWritingClean:
        num_lines = count_texts_of_output_individual(config_writing.config_writing_clean, "_clean")
    else:
        num_lines = count_texts_of_output_individual(config_writing)
    return num_lines


def count_texts_in_file(config):
    with open(config.file_path, "r") as f_in:
        func_reading = get_func_reading(config)
        i_text = 0
        for i_text, _ in func_reading(config, f_in):
            pass
    num_texts = i_text + 1
    return num_texts


def count_chars_in_text(config):
    with open(config.file_path, "r") as f_in:
        func_reading = get_func_reading(config)
        text_all = ""
        for _, text in func_reading(config, f_in):
            text_all += text
    return len(text_all)


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
    if type(config_reading) is ConfigReadingTxt and not config_reading.txt_has_lines:
        num_texts = count_chars_in_text(config_reading)
    else:
        num_texts = count_texts_in_file(config_reading)
    segment_start_end_list = create_segment_start_end_list_of_quantity(num_texts, num_segments)
    return segment_start_end_list


def create_percentage_segment_dict(i_start, i_end):
    percentage_segments = create_segment_start_end_list_of_quantity(i_end - i_start, 100)
    percentage_segments = [e + i_start - 1 for s,e in percentage_segments]
    percentage_segments_dict = {}
    for percentage, i_segment in enumerate(percentage_segments, start=1):
        percentage_segments_dict[i_segment] = percentage
    return percentage_segments_dict


def adapt_config_to_tmp(config_writing, config_processing, process_id):
    if config_processing.cpu_count > 1:
        config_writing_per_process = copy.copy(config_writing)
        if type(config_writing_per_process) is ConfigWritingClean:
            config_writing_per_process.config_writing_clean = copy.copy(config_writing.config_writing_clean)
            config_writing_per_process.config_writing_dirty = copy.copy(config_writing.config_writing_dirty)
            config_writing_per_process.config_writing_clean.file_path = (TMP_FOLDER + "tmp_clean_"
                + str(process_id) + "." 
                + get_filetype_of_config(config_writing.config_writing_clean))
            config_writing_per_process.config_writing_dirty.file_path = (TMP_FOLDER + "tmp_dirty_" 
                + str(process_id) + "." 
                + get_filetype_of_config(config_writing.config_writing_dirty))
        else:
            config_writing_per_process.file_path = (TMP_FOLDER + "tmp_" + str(process_id)
                + "." + get_filetype_of_config(config_writing))
    else:
        config_writing_per_process = config_writing
    return config_writing_per_process


def adapt_config_to_file_type(config_reading_or_writing):
    if type(config_reading_or_writing) is ConfigWritingClean:
        config_reading_or_writing.config_writing_clean = adapt_config_to_file_type(
            config_reading_or_writing.config_writing_clean
        )
        config_reading_or_writing.config_writing_dirty = adapt_config_to_file_type(
            config_reading_or_writing.config_writing_dirty
        )
    else:
        try:
            file_path = config_reading_or_writing.file_path
        except:
            pass
        try:
            file_path = config_reading_or_writing.file_path
        except:
            pass
        file_type = file_path.split(".")[-1]
        if file_type == "txt":
            if type(config_reading_or_writing) is ConfigReading:
                config_reading_or_writing = ConfigReadingTxt(**asdict(config_reading_or_writing))
                if config_reading_or_writing.txt_has_lines is None:
                    config_reading_or_writing.txt_has_lines = True
            elif type(config_reading_or_writing) is ConfigWriting:
                config_reading_or_writing = ConfigWritingTxt(**asdict(config_reading_or_writing))
    return config_reading_or_writing


def check_if_file_paths(config_reading, config_writing):
    if config_reading.file_path:
        if type(config_writing) is ConfigWritingClean:
            if config_writing.config_writing_clean.file_path:
                return  True
            else:
                return False
        else:
            if config_writing.file_path:
                return True
            else:
                return False
    else:
        return False


def processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
    func_reading = get_func_reading(config_reading)
    func_processing = get_func_processing(config_processing)
    percentage_segment_dict = None
    if not (type(config_reading) is ConfigReadingTxt and not config_reading.txt_has_lines):
        percentage_segment_dict = create_percentage_segment_dict(start_end_segment[0], start_end_segment[1])
    for i_text, text in func_reading(config_reading, f_in, start_end_segment):
        for text_processed in func_processing(config_processing, text):
            if percentage_segment_dict and i_text in percentage_segment_dict:
                percentage = percentage_segment_dict.get(i_text)
                print(f"process_id: {process_id}: {percentage}%")
            yield text_processed


def processing_context_common(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with open(config_reading.file_path, "r") as f_in, open(config_writing.file_path, "w") as f_out:
        func_writing = get_func_writing(config_writing)
        for text_processed in processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
            func_writing(config_writing, text_processed, f_out)


def processing_context_clean(config_processing, config_reading, config_writing, process_id, start_end_segment):
    with (
        open(config_reading.file_path, "r") as f_in, 
        open(config_writing.config_writing_clean.file_path, "w") as f_out_clean,
        open(config_writing.config_writing_dirty.file_path, "w") as f_out_dirty,
    ):
        func_writing_clean = get_func_writing(config_writing.config_writing_clean)
        func_writing_dirty = get_func_writing(config_writing.config_writing_dirty)
        for text_processed, is_text_processed_clean in processing_execution(config_processing, config_reading, process_id, start_end_segment, f_in):
            if is_text_processed_clean:
                func_writing_clean(config_writing.config_writing_clean, text_processed, f_out_clean)
            else:
                func_writing_dirty(config_writing.config_writing_dirty, text_processed, f_out_dirty)


def main_process_multi(config_processing, config_reading, config_writing):
    DEBUG_SINGLE_PROCESS = False
    print("- all processing start ----------------------------------------------")
    segment_start_end_list = create_segment_start_end_list_of_file(
        config_reading, 
        config_processing.cpu_count
    )
    func_context_processing = get_func_context_processing(config_processing)
    process_list = []
    for process_id, segment_start_end in enumerate(segment_start_end_list):
        config_writing_per_process = adapt_config_to_tmp(config_writing, config_processing, process_id)
        print(f"- process_id {process_id}: start -----------------------------------------------")
        if DEBUG_SINGLE_PROCESS:
            func_context_processing(
                config_processing,
                config_reading,
                config_writing_per_process,
                process_id, 
                segment_start_end, 
            )
        else:
            process = Process(
                target=func_context_processing,
                args=(
                    config_processing,
                    config_reading,
                    config_writing_per_process,
                    process_id, 
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
    config_reading = create_config_reading()
    config_writing = create_config_writing()
    config_writing_metadata = create_config_writing_metadata()
    config_processing = create_config_processing()

    # config adaptions and calling into main_process_multi
    if check_if_file_paths(config_reading, config_writing):
        config_reading = adapt_config_to_file_type(config_reading)
        config_writing = adapt_config_to_file_type(config_writing)
        main_process_multi(config_processing, config_reading, config_writing)
    else:
        for file in os.listdir(config_reading.folder):
            if not(config_reading.ignore_file and file in config_reading.ignore_file):
                print(f"processing file: {file}")
                config_reading.file_path = config_reading.folder + file
                if type(config_writing) is ConfigWritingClean:
                    file_split = file.split(".")
                    file_name = "".join(file_split[:-1])
                    file_type = file_split[-1]
                    if config_processing.append_to_file:
                        file_name_clean = file_name + "_clean"
                        file_name_dirty = file_name + "_dirty"
                    else:
                        file_name_clean = file_name
                        file_name_dirty = file_name
                    config_writing.config_writing_clean.file_path = (
                        config_writing.config_writing_clean.folder + file_name_clean + "." 
                        + file_type)
                    config_writing.config_writing_dirty.file_path = (
                        config_writing.config_writing_dirty.folder + file_name_dirty + "." 
                        + file_type)
                else:
                    config_writing.file_path = config_writing.folder + file
                config_reading = adapt_config_to_file_type(config_reading)
                config_writing = adapt_config_to_file_type(config_writing)
                main_process_multi(config_processing, config_reading, config_writing)
                config_reading.file_path = None
                if type(config_writing) is ConfigWritingClean:
                    config_writing.config_writing_clean.file_path = None
                    config_writing.config_writing_dirty.file_path = None

    # write metadata
    if config_writing_metadata:
        write_veld_data_yaml(config_writing_metadata, config_writing)


if __name__ == "__main__":
    main()

