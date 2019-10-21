/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 20, 2019
 *
 * mapping.hpp
 **/

#ifndef MAPPING_HPP
#define MAPPING_HPP

typedef struct mapping
{
    float x, y; // translation vertex
    float a, b, c, d; // scaling/rotation matrix
    float p; // mapping probability
}
mapping;

#endif
