import os

this_base_directory = "/path/including/mapping_llm_political_space"
cache_base_directory = "/your/dir/where/hf_cache/is"

os.environ["HF_HOME"] = cache_base_directory + "hf_cache" 
os.environ["HF_MODEL_CACHE"] = cache_base_directory + "hf_cache"
os.environ["HF_DATASETS_CACHE"] = cache_base_directory + "hf_cache"
os.environ["HF_TOKEN"] = "hf_yourtoken"
os.environ["OPENAI_API_KEY"] = "t438w9th945samaltmanwillkillusall3q89g53"

do_tqdm=True #useful to disable when you are piping stdout into a log file or working via slurm