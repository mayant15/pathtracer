#pragma once

#include <cmath>
#include <vector>
#include <random>

//*****************************************************************************
// GEOMETRIC AND MATH CLASSES
//*****************************************************************************

#define M_PI 3.14159265359
#define M_INF 1e20
#define M_EPSILON 1e-4

double sample() {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distr(0.0,1.0);
    return distr(generator);
}

/*! \brief Clamp the provided value between 0 and 1 */
inline double clamp(double x, double min = 0, double max = 1)
{ return x < min ? min : x > max ? max : x; }

struct Vec
{
    double x, y, z;

    explicit Vec(double x_ = 0, double y_ = 0, double z_ = 0)
            : x(x_), y(y_), z(z_)
    {}

    Vec operator+(const Vec& b) const
    { return Vec(x + b.x, y + b.y, z + b.z); }

    Vec operator-(const Vec& b) const
    { return Vec(x - b.x, y - b.y, z - b.z); }

    [[nodiscard]] double sqlen() const
    { return x * x + y * y + z * z; }

    [[nodiscard]] double len() const
    { return std::sqrt(sqlen()); }

    /*! \brief Normalize the vector in-place */
    Vec& norm()
    { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }

    // Scalar multiplication
    Vec operator*(double b) const
    { return Vec(x * b, y * b, z * b); }

    // Component-wise multiplication
    Vec operator*(const Vec& b) const
    { return Vec(x * b.x, y * b.y, z * b.z); }

    // Dot product
    [[nodiscard]] double dot(const Vec& b) const
    { return x * b.x + y * b.y + z * b.z; } // cross:

    // Cross product
    Vec operator%(const Vec& b) const
    { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }

    void clamp()
    {
        x = ::clamp(x);
        y = ::clamp(y);
        z = ::clamp(z);
    }
};

struct Basis
{
    Vec u, v, w;
};

struct Mat3
{
    Vec operator*(const Vec& v)
    {
        return Vec(
                Vec(_data[0], _data[1], _data[2]).dot(v),
                Vec(_data[3], _data[4], _data[5]).dot(v),
                Vec(_data[6], _data[7], _data[8]).dot(v)
        );
    }

    double& at(size_t row, size_t col)
    {
        return _data[row * col];
    }

private:
    double _data[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
};

struct Ray
{
    Vec origin;
    Vec direction;

    Ray(Vec origin_, Vec direction_)
            : origin(origin_), direction(direction_)
    {}
};

// material types, used in radiance()
enum class ESurfaceType
{
    DIFFUSE,
    SPECULAR,
    REFRACTIVE
};

struct Sphere
{
    double radius;
    Vec position;
    Vec emission;
    Vec color;
    ESurfaceType surface;

    Sphere(double rad_, Vec p_, Vec e_, Vec c_, ESurfaceType surface_)
            : radius(rad_),
              position(p_),
              emission(e_),
              color(c_),
              surface(surface_)
    {}

    [[nodiscard]] double intersect(const Ray& ray) const
    {
        // For intersection, solve
        // |(o + t*dir) - position| = radius
        // (op + t * dir).(op + t * dir) = radius^2
        // (dir.dir)t^2 + 2(op.dir)t + op.op - radius^2 = 0
        // i.e solve At^2 + Bt + C = 0

        Vec op = ray.origin - position;
        double A = ray.direction.sqlen();
        double B = 2 * op.dot(ray.direction);
        double C = op.sqlen() - radius * radius;

        double D = B * B - 4 * A * C;
        if (D < 0) return 0; // no solution
        else D = std::sqrt(D);

        double t, eps = 1e-4;
        double t1 = (-1 * B + D) / (2 * A);
        double t2 = (-1 * B - D) / (2 * A);

        // returns distance, 0 if nohit
        if (t1 > M_EPSILON && t1 < t2) return t1;
        else if (t2 > M_EPSILON) return t2;
        else return 0;
    }
};

/*! Change float color representation to integer representation */
inline int to_int(double x)
{
    double gamma_adjust = std::pow(clamp(x), 1 / 2.2);
    return std::floor(gamma_adjust * 255 + .5);
}

inline Vec reflect(const Vec& incident, const Vec& normal)
{
    return incident - normal * 2 * normal.dot(incident);
}
