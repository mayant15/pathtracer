// smallpt, a Path Tracer by Kevin Beason, 2008
// Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
//        Remove "-fopenmp" for g++ version < 4.2
// Usage: time ./smallpt 5000 && xv image.ppm

#include <iostream>
#include <stb_image_write.h>
#include "integrators.h"

constexpr unsigned int WIDTH = 1024;
constexpr unsigned int HEIGHT = 768;
constexpr double ASPECT_RATIO = (double) WIDTH / HEIGHT;
constexpr int NR_SAMPLES = 1;

int main()
{
    Camera camera;
    camera.position = Vec (0, 4, 8);
    camera.look_at = Vec (0, 4, -8);
    camera.fov = 45;

    auto basis = camera.get_basis(ASPECT_RATIO);

    Scene scene = Scene::cornell_box();

    Integrator integrator;
    integrator.scene = &scene;

    std::vector<uint32_t> img_data(WIDTH * HEIGHT);

#pragma omp parallel for
    for (unsigned short row = 0; row < HEIGHT; ++row)
    {
        // Loop over image rows
        std::printf("\nRendering (%d spp) %5.2f%%", NR_SAMPLES * 4, 100. * row / (HEIGHT - 1));

        // Column loop
        for (unsigned short col = 0; col < WIDTH; ++col)
        {
            Vec color;

            // 2x2 Sub-sample loop
            for (int sy = 0; sy < 2; ++sy) // 2x2 subpixel rows
            {
                for (int sx = 0; sx < 2; ++sx) // 2x2 subpixel cols
                {
                    for (int s = 0; s < NR_SAMPLES; s++)
                    {
                        // Sample a circle
                        double r1 = 2 * sample(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * sample(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                        double x_extent = ((col + (sx + dx) / 2) / (double) WIDTH) * 2 - 1;
                        double y_extent = ((row + (sy + dy) / 2) / (double) HEIGHT) * 2 - 1;

                        // Get Ray for pixel
                        Ray ray(camera.position, (basis.u * x_extent + basis.v * y_extent + basis.w).norm());
                        color = color + integrator.integrate(ray, 0);
                    }
                }
            }

            color = color * (1. / (NR_SAMPLES * 4));
            color.clamp();

            img_data[row * WIDTH + col]
                    = (0xFF << 24)              /* alpha */
                      | (to_int(color.z) << 16) /* blue */
                      | (to_int(color.y) << 8)  /* green */
                      | (to_int(color.x));      /* red */
        }
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_png("image.png", WIDTH, HEIGHT, 4, img_data.data(), WIDTH * sizeof(uint32_t));
}
