#ifndef UNICODE_IO_H
#define UNICODE_IO_H

#include <string>

#include <opencv2/opencv.hpp>

std::wstring utf8_to_wstring(const std::string& text);
std::string wstring_to_utf8(const std::wstring& text);

cv::Mat imread_unicode(const std::wstring& path, int flags);
bool imwrite_unicode(const std::wstring& path, const cv::Mat& image);

#endif
