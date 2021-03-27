#pragma once

#include "math_types.h"

const Vec UP = Vec(0, 1, 0);

struct Camera
{
    Vec position;
    Vec look_at;
    double fov;

    Basis get_basis(double aspect_ratio) const
    {
        // Camera space basis (u, v, w)
        const Vec w = look_at - position;
        double l = w.len() * std::tan(fov * M_PI / 180.);
        const Vec u = (w % UP).norm() * l;
        const Vec v = (u % w).norm() * (l / aspect_ratio);

        return { u, v, w };
    }
};

struct Scene
{
    std::vector<Sphere> objects;

    bool intersect(const Ray& ray, double& out_t, int& out_id)
    {
        out_t = M_INF;
        for (int i = objects.size() - 1; i >= 0; --i)
        {
            auto d = objects[i].intersect(ray);
            if (d > 0 && d < out_t)
            {
                out_t = d;
                out_id = i;
            }
        }
        return out_t < M_INF;
    }

    Sphere& get_object(size_t id)
    {
        return objects[id];
    }

    static Scene cornell_box()
    {
        Scene s;
        s.objects = {
                // Scene: radius, position, emission, color, material
                /* Left */   Sphere(1e5, Vec(-1e5 - 5, 0, 0), Vec(), Vec(.75, .25, .25), ESurfaceType::DIFFUSE),
                /* Right */  Sphere(1e5, Vec(1e5 + 5, 0, 0), Vec(), Vec(.25, .25, .75), ESurfaceType::DIFFUSE),
                /* Back */   Sphere(1e5, Vec(0, 0, -1e5 - 5), Vec(), Vec(.75, .75, .75), ESurfaceType::DIFFUSE),
                /* Bottom */ Sphere(1e5, Vec(0, -1e5, 0), Vec(), Vec(.75, .75, .75), ESurfaceType::DIFFUSE),
                /* Top */    Sphere(1e5, Vec(0, 1e5 + 10, 0), Vec(), Vec(.75, .75, .75), ESurfaceType::DIFFUSE),
                /* Mirror */ Sphere(1.5, Vec(-2, 2, -2), Vec(), Vec(1, 1, 1) * .999, ESurfaceType::SPECULAR),
                /* Glass */  Sphere(1.5, Vec(2, 2, 1), Vec(), Vec(1, 1, 1) * .999, ESurfaceType::REFRACTIVE),
                /* Light */  Sphere(1, Vec(0, 10.5, 0), Vec(12, 12, 12), Vec(), ESurfaceType::DIFFUSE)
        };
        return s;
    }

    static Scene normal()
    {
        Scene s;
        s.objects = {
                Sphere(1, Vec(0, 1.5, 0), Vec(), Vec(1, 0.1, 0.1), ESurfaceType::SPECULAR),
                Sphere(1e5, Vec(0, -1e5, 0), Vec(), Vec(0.5, 1.0, 0.2), ESurfaceType::DIFFUSE)
        };
        return s;
    }
};
