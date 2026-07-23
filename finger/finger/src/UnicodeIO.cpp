#include "UnicodeIO.h"

#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// 本文件负责 Windows 下的 Unicode 路径读写兼容，
// 避免中文路径或宽字符路径导致 OpenCV 直接 imread / imwrite 失败。
std::wstring utf8_to_wstring(const std::string& text) {
#ifdef _WIN32
    if (text.empty()) return std::wstring();
    const int len = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring result(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, result.data(), len);
    return result;
#else
    return std::wstring(text.begin(), text.end());
#endif
}

std::string wstring_to_utf8(const std::wstring& text) {
#ifdef _WIN32
    if (text.empty()) return std::string();
    const int len = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, result.data(), len, nullptr, nullptr);
    return result;
#else
    return std::string(text.begin(), text.end());
#endif
}

cv::Mat imread_unicode(const std::wstring& path, int flags) {
    // Windows 下先把文件按二进制方式读入，再交给 imdecode，
    // 可以绕开 OpenCV 对宽字符路径支持不稳定的问题。
#ifdef _WIN32
    std::ifstream file(path, std::ios::binary);
    if (!file) return cv::Mat();
    std::vector<uchar> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (buffer.empty()) return cv::Mat();
    return cv::imdecode(buffer, flags);
#else
    return cv::imread(std::string(path.begin(), path.end()), flags);
#endif
}

bool imwrite_unicode(const std::wstring& path, const cv::Mat& image) {
    // 写文件同理：先编码到内存，再自己写到宽字符路径。
#ifdef _WIN32
    const std::string utf8_path = wstring_to_utf8(path);
    std::string ext = ".png";
    const std::size_t dot_pos = utf8_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = utf8_path.substr(dot_pos);
    }

    std::vector<uchar> encoded;
    if (!cv::imencode(ext, image, encoded)) return false;

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    return file.good();
#else
    return cv::imwrite(std::string(path.begin(), path.end()), image);
#endif
}
