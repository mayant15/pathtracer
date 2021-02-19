#pragma once

#include "launch_params.h"

template<typename T>
struct sbt_record_t
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using ray_gen_sbt_record_t = sbt_record_t<ray_gen_data_t>;
using miss_sbt_record_t = sbt_record_t<int>;
