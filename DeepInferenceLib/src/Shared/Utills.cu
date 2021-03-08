#ifndef UTILLS_CU
#define UTILLS_CU

#include "Utills.cuh"
#include <algorithm> 
#include <cctype>
#include <locale>

LayerType Utills::Layers::getLayerType(const Layer* layer)
{
    LayerType type;
    const std::type_info& typeInfoOfLayer = typeid(*layer);
    if (typeInfoOfLayer == typeid(ConvLayer))
        type = LayerType::CONV;
    if (typeInfoOfLayer == typeid(Flatten))
        type = LayerType::FLATTEN;
    if (typeInfoOfLayer == typeid(DenseLayer))
        type = LayerType::DENSE;
    return type;
}

void Utills::StringUtils::ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
        }));
}

void Utills::StringUtils::rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
        }).base(), s.end());
}

void Utills::StringUtils::trim(std::string& s) {
    Utills::StringUtils::ltrim(s);
    rtrim(s);
}

std::string Utills::StringUtils::ltrim_copy(std::string s) {
    Utills::StringUtils::ltrim(s);
    return s;
}

std::string Utills::StringUtils::rtrim_copy(std::string s) {
    Utills::StringUtils::rtrim(s);
    return s;
}

std::string Utills::StringUtils::trim_copy(std::string s) {
    Utills::StringUtils::trim(s);
    return s;
}

#endif // !UTILLS_CU
