#pragma once

#include <iostream>

#define LOG_LEVEL_INFO
#define LOG_LEVEL_ERROR


#ifdef LOG_LEVEL_INFO
#define LOG_INFO(...) fprintf(stdout, __VA_ARGS__)
#else
#define LOG_INFO(...)
#endif

#ifdef LOG_LEVEL_ERROR
#define LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#else
#define LOG_ERROR(...)
#endif
