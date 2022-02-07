#pragma once
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#ifdef USING_BOOST
#include <boost/algorithm/string.hpp>
#else
#include "str_split.h"
#endif

// To read more about boost string algorithm:  https://theboostcpplibraries.com/boost.stringalgorithms

/*
 * A class to read data from a csv file.
 */
class CSVReader
{
    std::string fileName;
    std::string delimeter;

public:
    CSVReader(std::string filename, std::string delm = ",") :
        fileName(filename), delimeter(delm)
    { }

    // Function to fetch data from a CSV File
    std::vector<std::vector<std::string>> getData();
};

/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string>> CSVReader::getData()
{
    std::ifstream file(fileName);

    std::vector<std::vector<std::string>> data_list;
   
    std::string line;
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        std::vector<std::string> vec;
		#ifdef USING_BOOST
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		#else
        vec = str_util::split(line, delimeter);
		#endif
        data_list.push_back(vec);
    }

    // Close the File
    file.close();

    return data_list;
}
