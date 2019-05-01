import json
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

SESSION_KEY = ""


def read_config():
    with open(PROJECT_DIR + '/coogleconfig.json') as config_json:
        config_data = json.load(config_json)
    return config_data

def get_rawfeature_extractor_loc():

    config_data = read_config()
    return config_data["rawfeatureextractor"]["loc"]
def get_idaloc():
	config_data = read_config()
	return config_data["rawfeatureextractor"]["idaloc"]


def get_embedding_loc():
	config_data = read_config()
	return config_data["embedding"]["loc"]
#test

def get_feature_vector_size():
	config_data = read_config()
	return config_data["embedding"]["FEATURE_VECTOR_SIZE"]




def get_lsh_cache_random_seed():
	config_data = read_config()
	return config_data["searchengine"]["LSH_CACHE_RANDOM_SEED"]


def get_lsh_projection_len():
	config_data = read_config()
	return config_data["searchengine"]["LSH_PROJECTION_LEN"]

def get_lsh_result_num():
	config_data = read_config()
	return config_data["searchengine"]["LSH_RESULT_NUM"]

def get_lsh_enable_multi_process_search():
	config_data = read_config()
	return config_data["searchengine"]["ENABLE_MULTI_PROCESS_SEARCH"]


	'''


"database":{
	"DB_IP":"10.8.0.1",
	"DB_PORT":27017,
	"DB_USER_NAME":"coogletest",
	"DB_USER_PASSWD":"deepbits"
}

	'''


def get_db_ip():
	config_data = read_config()
	return config_data["database"]["DB_IP"]

def get_db_port():
	config_data = read_config()
	return config_data["database"]["DB_PORT"]

def get_db_username():
	config_data = read_config()
	return config_data["database"]["DB_USER_NAME"]

def get_db_passwd():
	config_data = read_config()
	return config_data["database"]["DB_USER_PASSWD"]
	


def get_tempfolder():
	config_data = read_config()
	tempfolder = config_data["tempfile"]["tempfolder"]
	if not os.path.isdir(tempfolder):
		os.makedirs(tempfolder)
	return tempfolder
def get_fdfs_conf():
	config_data = read_config()
	return config_data["fdfs"]["conf_loc"]


def get_fdfs_loc():
	config_data = read_config()
	return config_data["fdfs"]["fdfs_loc"]

if __name__ == '__main__':

    print(get_rawfeature_extractor_loc())
    print(get_embedding_loc())
    print( get_lsh_cache_random_seed())
    print( get_lsh_projection_len())
    print( get_lsh_result_num())
    print( get_lsh_enable_multi_process_search())
    print( get_idaloc())
    print( get_fdfs_conf())
    print( get_tempfolder())
    print(get_fdfs_loc())