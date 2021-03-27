#pragma once

#include "math_types.h"
#include "scene.h"

struct Integrator
{
    // (0.527, 0.804, 0.917) is light blue
    const Vec bg_color = Vec (0.2, 0.2, 0.2);
    Scene* scene;

    Vec integrate(Ray ray, int depth)
    {
        if (depth >= 3) return bg_color;

        double t = 0;
        int id = 0;
        if (!scene->intersect(ray, t, id)) return bg_color;

// f = color
// p = max out of RGB
//    if (++depth > 5) if (erand48(Xi) < p) f = f * (1 / p); else return obj.e; // Russian Roulette

        const Sphere& sp = scene->get_object(id); // spheres[id];
        Vec hit = ray.origin + ray.direction * t;
        Vec normal = (hit - sp.position).norm();

        if (sp.surface == ESurfaceType::DIFFUSE)
        {
            // Sampling the hemisphere
            double r1 = 2 * M_PI * sample();
            double r2 = sample();
            double r2s = sqrt(r2);

            // Basis
            Vec w = normal;
            Vec u = (w % UP).norm();
            Vec v = (u % w).norm();

            // Random ray direction
            Vec dir = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
            return sp.emission + sp.color * integrate(Ray(hit, dir), depth + 1);
        }
        else if (sp.surface == ESurfaceType::SPECULAR)
        {
            Ray refl(hit, reflect(ray.direction, normal));
            return sp.emission + sp.color * integrate(refl, depth + 1);
        }
        else if (sp.surface == ESurfaceType::REFRACTIVE)
        {
            Ray refl(hit, reflect(ray.direction, normal));

            Vec nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;
            bool into = normal.dot(nl) > 0; // Ray from outside going in?
            double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = ray.direction.dot(nl), cos2t;

            // Total internal reflection
            if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)
            {
                return sp.emission + sp.color * integrate(refl, depth + 1);
            }

            Vec tdir = (ray.direction * nnt - normal * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
            double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(normal));
            double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);

            return sp.emission
                   + sp.color * (depth > 2 ? (sample() < P ?    // Russian roulette
                                              integrate(refl, depth) * RP : integrate(Ray(hit, tdir), depth) * TP)
                                           : integrate(refl, depth) * Re + integrate(Ray(hit, tdir), depth) * Tr);
        }
        else
        {
            return bg_color;
        }
    }
};
