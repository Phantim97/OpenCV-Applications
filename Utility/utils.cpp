#include <filesystem>
#include <iostream>
#include "utils.h"
void read_file_names(const std::string& dir, std::vector<std::string>& file_vector)
{
	//Get all filenames in directory
	for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dir))
	{
		file_vector.push_back(entry.path().string());
	}
}

void list_files_in_directory(const std::string& dir_name)
{
	std::vector<std::string> files;
	read_file_names(dir_name, files);

	for (int i = 0; i < files.size(); i++)
	{
		std::cout << files[i] << '\n';
	}
}