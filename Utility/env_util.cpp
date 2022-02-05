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

	std::string get_dataset_path()
	{
		std::string path = getenv("DATASET_PATH");
		path += '/';
		return path;
	}

	std::string get_tessdata_path()
	{
		std::string path = getenv("LIBS_PATH");
		path += "/tesseract/tessdata";
		return path;
	}
}