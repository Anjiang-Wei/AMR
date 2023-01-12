
#ifndef _LEGION_HOOK_H
#define _LEGION_HOOK_H

#include "legion.h"

typedef Legion::Rect <1, long long int> Box1D;
typedef Legion::Rect <2, long long int> Box2D;
typedef Legion::Rect <3, long long int> Box3D;

typedef Legion::Point<1, long long int> Point1D;
typedef Legion::Point<2, long long int> Point2D;
typedef Legion::Point<3, long long int> Point3D;

typedef Legion::IndexSpace       IndexSpece;
typedef Legion::IndexPartition   IndexPartition;
typedef Legion::Context          Context;
typedef Legion::Runtime          Runtime;
typedef Legion::PhysicalRegion   PhysicalRegion;
typedef Legion::LogicalRegion    LogicalRegion;
typedef Legion::LogicalPartition LogicalPartition;
typedef Legion::DomainPoint      DomainPoint;
typedef Legion::Domain           Domain;
typedef Legion::Task             Task;
typedef Legion::FieldSpace       FieldSpace;
typedef Legion::FieldID          FieldID;
typedef Legion::TaskArgument     TaskArguent;

#endif
