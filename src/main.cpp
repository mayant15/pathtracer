#include "types.h"
#include "checkers.h"
#include "renderer.h"

#include <stb_image_write.h>

#include <stdexcept>

constexpr unsigned int IMAGE_WIDTH = 1024;
constexpr unsigned int IMAGE_HEIGHT = 728;
constexpr unsigned int BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr unsigned int BUFFER_SIZE_IN_BYTES = sizeof(color_t) * BUFFER_SIZE;

void write_image(const std::vector<color_t>& data, const std::string& path, uint2 dims)
{
    stbi_flip_vertically_on_write(1);
    stbi_write_png(path.c_str(), dims.x, dims.y, 4, data.data(), dims.x * sizeof(color_t));
}

int main(int argc, char* argv[])
{
    try
    {
        // Initialize renderer
        render_options_t opt {};
        opt.height = IMAGE_HEIGHT;
        opt.width = IMAGE_WIDTH;
        renderer_t renderer { opt };

        // Load the scene
        scene_t scene("path-to-scene-desc.scene");
        renderer.load_scene(scene);

        // Allocate device storage for image
        device_buffer_t buffer { BUFFER_SIZE_IN_BYTES };
        renderer.render(buffer);

        // Copy result back to host
        std::vector<color_t> result(BUFFER_SIZE);
        buffer.fetch(result.data(), BUFFER_SIZE_IN_BYTES);

        // Save image to disk
        write_image(result, "out.png", { IMAGE_WIDTH, IMAGE_HEIGHT });

        renderer.cleanup();
    }
    catch (std::exception& e)
    {
        LOG_ERROR("ERROR: %s\n", e.what());
        exit(1);
    }
    return 0;
}
