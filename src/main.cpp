#include "types.h"
#include "checkers.h"
#include "renderer.h"
#include "image.h"

#include <stdexcept>

using color_t = uchar4;

constexpr unsigned int IMAGE_WIDTH = 1024;
constexpr unsigned int IMAGE_HEIGHT = 728;
constexpr unsigned int BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr unsigned int BUFFER_SIZE_IN_BYTES = sizeof(color_t) * BUFFER_SIZE;

int main(int argc, char* argv[])
{
    try
    {
        render_options_t opt {};
        opt.height = IMAGE_WIDTH;
        opt.width = IMAGE_HEIGHT;

        renderer_t renderer { opt };

        // Allocate device storage for image
        device_buffer_t buffer { BUFFER_SIZE_IN_BYTES };
        renderer.render(buffer);

        // Copy result back to host
        std::vector<uchar4> result(BUFFER_SIZE);
        buffer.fetch(result.data(), buffer.size());

        // Save image to disk
        write_image(result, "out.png", { IMAGE_WIDTH, IMAGE_HEIGHT} );

        renderer.cleanup();
    }
    catch (std::exception& e)
    {
        LOG_ERROR("ERROR: %s\n", e.what());
        exit(1);
    }
    return 0;
}
