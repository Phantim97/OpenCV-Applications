// label data
// labelData.h
#pragma once
#include <iostream>
#include <string>
#include <map>

// Label -> Name Mapping file
typedef std::map<std::string, std::string> Dict;
Dict generateLabelMap(void);