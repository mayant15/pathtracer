#pragma once

#include <fstream>
#include <string>

static const std::string PTX_PATH_STRING = PTX_PATH;

std::string load_ptx_string(const std::string& path)
{
    std::ifstream file { PTX_PATH_STRING + path };
    std::string content { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
    return content;
}
