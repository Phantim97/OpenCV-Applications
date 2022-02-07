#pragma once
//replicated version of boost::algorithm::split
//From https://github.com/trikitrok/StringCalculatorAdditionUsingGoogleMock/tree/master/code

#include <regex>
#include <vector>

namespace VectorUtils
{
    template<typename T, class UnaryPredicate>
    std::vector<T> filter(const std::vector<T>& original, UnaryPredicate pred) {

        std::vector<T> filtered;

        std::copy_if(begin(original), end(original),
            std::back_inserter(filtered),
            pred);

        return filtered;
    }

    template<typename T2, typename T1, class UnaryOperation>
    std::vector<T2> map(const std::vector<T1>& original, UnaryOperation mappingFunction) {

        std::vector<T2> mapped;

        std::transform(begin(original), end(original),
            std::back_inserter(mapped),
            mappingFunction);

        return mapped;
    }

    template<typename T>
    void append(std::vector<T>& appendedTo, const std::vector<T>& appended) {
        appendedTo.insert(end(appendedTo), begin(appended), end(appended));
    }
}

namespace str_util
{
    std::string escapeChar(char character)
    {
        const std::unordered_map<char, std::string> ScapedSpecialCharacters = {
          {'.', "\\."}, {'|', "\\|"}, {'*', "\\*"}, {'?', "\\?"},
          {'+', "\\+"}, {'(', "\\("}, {')', "\\)"}, {'{', "\\{"},
          {'}', "\\}"}, {'[', "\\["}, {']', "\\]"}, {'^', "\\^"},
          {'$', "\\$"}, {'\\', "\\\\"}
        };

        std::unordered_map<char, std::string>::const_iterator it = ScapedSpecialCharacters.find(character);

        if (it == ScapedSpecialCharacters.end())
        {
            return std::string(1, character);
        }

        return it->second;
    }

    std::string escapeString(const std::string& str)
    {
        std::stringstream stream;
        std::for_each(begin(str), end(str),
            [&stream](const char character) { stream << escapeChar(character); }
        );
        return stream.str();
    }

    std::vector<std::string> escapeStrings(
        const std::vector<std::string>& delimiters)
    {
        return VectorUtils::map<std::string>(delimiters, escapeString);
    }

    bool isAnInteger(const std::string& token) {
        const std::regex e("\\s*[+-]?([1-9][0-9]*|0[0-7]*|0[xX][0-9a-fA-F]+)");
        return std::regex_match(token, e);
    }

    std::string extractRegion(const std::string& str,
        int from, int to)
    {
        std::string region = "";
        int regionSize = to - from;
        return str.substr(from, regionSize);
    }

    int convertToInt(const std::string& str)
    {
        std::string::size_type sz;
        return std::stoi(str, &sz);
    }

    std::string join(const std::vector<std::string>& tokens, const std::string& delimiter)
    {
        std::stringstream stream;

        stream << tokens.front();

        std::for_each(begin(tokens) + 1, end(tokens),
            [&](const std::string& elem)
            {
                stream << delimiter << elem;
            }
        );

        return stream.str();
    }

    std::vector<std::string> split(
        const std::string& str,
        const std::vector<std::string>& delimiters)
    {

        std::regex rgx(str_util::join(escapeStrings(delimiters), "|"));

        std::sregex_token_iterator
            first{ begin(str), end(str), rgx, -1 },
            last;

        return{ first, last };
    }

    std::vector<std::string> split(const std::string& str,
        const std::string& delimiter)
    {
        std::vector<std::string> delimiters = { delimiter };
        return split(str, delimiters);
    }
}
