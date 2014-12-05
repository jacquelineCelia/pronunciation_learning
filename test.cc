#include <iostream>

#include "randomizer.h"

using namespace std;

int main() {
    Randomizer r;
    r.SetDataSize(105);
    r.SetUpBatchIndices(25);
    for (int i = 0; i < 10; ++i) {
        cerr << "starting: " << r.GetStartIndex(i) << endl;
        cerr << "ending: " << r.GetEndIndex(i) << endl;
    }
    r.SetUpBatchIndices(35);
    for (int i = 0; i < 10; ++i) {
        cerr << "starting: " << r.GetStartIndex(i) << endl;
        cerr << "ending: " << r.GetEndIndex(i) << endl;
    }
    r.SetUpBatchIndices(135);
    for (int i = 0; i < 10; ++i) {
        cerr << "starting: " << r.GetStartIndex(i) << endl;
        cerr << "ending: " << r.GetEndIndex(i) << endl;
    }
    r.SetUpBatchIndices(105);
    for (int i = 0; i < 10; ++i) {
        cerr << "starting: " << r.GetStartIndex(i) << endl;
        cerr << "ending: " << r.GetEndIndex(i) << endl;
    }
    return 0;
}
