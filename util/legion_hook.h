
#ifndef _LEGION_HOOK_H
#define _LEGION_HOOK_H

#include "legion.h"
#include <map>

typedef Legion::Rect <1, long long int> Box1D;
typedef Legion::Rect <2, long long int> Box2D;
typedef Legion::Rect <3, long long int> Box3D;

typedef Legion::Point<1, long long int> Point1D;
typedef Legion::Point<2, long long int> Point2D;
typedef Legion::Point<3, long long int> Point3D;

typedef Legion::IndexSpace          IndexSpace;
typedef Legion::IndexPartition      IndexPartition;
typedef Legion::Context             Context;
typedef Legion::Runtime             Runtime;
typedef Legion::PhysicalRegion      PhysicalRegion;
typedef Legion::LogicalRegion       LogicalRegion;
typedef Legion::LogicalPartition    LogicalPartition;
typedef Legion::DomainPoint         DomainPoint;
typedef Legion::Domain              Domain;
typedef Legion::Task                Task;
typedef Legion::Future              Future;
typedef Legion::RegionRequirement   RegionRequirement;
typedef Legion::CopyLauncher        CopyLauncher;
typedef Legion::AttachLauncher      AttachLauncher;
typedef Legion::FieldSpace          FieldSpace;
typedef Legion::FieldID             FieldID;
typedef Legion::TaskArgument        TaskArguent;
typedef Legion::FieldAllocator      FieldAllocator;

typedef Legion::Transform<2,2,Legion::coord_t>  Transform2D;

typedef Legion::PointInRectIterator<1> PointInBox1D;
typedef Legion::PointInRectIterator<2> PointInBox2D;
typedef Legion::PointInRectIterator<3> PointInBox3D;

typedef Legion::IndexLauncher IndexLauncher;
typedef Legion::TaskArgument  TaskArgument;
typedef Legion::ArgumentMap   ArgumentMap;

template<legion_privilege_mode_t privil, typename dtype, int dim> using FieldAccessor = Legion::FieldAccessor<privil, dtype, dim>;

#endif
