#include <iostream>
#include <string>

namespace util
{
	std::string get_model_path()
	{
		std::string path = getenv("MODELS_PATH");
		path += '/';
		return path;
	}

	std::string get_data_path()
	{
		std::string path = getenv("DATA_PATH");
		path += '/';
		return path;
	}
}