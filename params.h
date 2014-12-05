#include <vector>

#include "model.h"
#include "sampler.h"
#include "counter.h"
#include "datum.h"

struct Params {
    Model* model;
    Sampler* sampler;
    Counter* counter;
    std::vector<Datum*> data;
};
