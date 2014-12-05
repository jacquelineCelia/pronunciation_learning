#include <iostream>
#include <string>
#include <cstdlib>

#include "config.h"
#include "manager.h"

using namespace std;

void usage();

int main(int argc, char* argv[]) {
    if (argc != 21 && argc != 19) {
        usage();
        return -1;
    }
    int mode = atoi(argv[2]);
    string fn_list = argv[4];
    string fn_config = argv[6];
    int n_iter = atoi(argv[8]);
    string fn_gaussian = argv[10];
    string fn_sil = argv[12];
    string basedir = argv[14];
    int batch_size = atoi(argv[16]);
    string fn_gseed = argv[18];
    string fn_snapshot = "";

    if (argc == 21) {
        fn_snapshot = argv[20];
    }

    if (batch_size > 32767) {
        cout << "Batch size is beyond the randomization capability" << endl;
        exit(0);
    }
    Config config;
    if (!config.Load(fn_config, fn_gaussian)) {
        cout << "Cannot load configuration file." 
            << " Check " << fn_config << endl;
    }
    else {
        cout << "Configuration file loaded successfully." << endl;
    }
    // config.print();
    Manager manager(&config);
    if (mode == 0) {
        if (fn_snapshot == "") {
            cout << "No file model is specified. Need a previous snapshot file" << endl;
        }
        else {
            manager.InitializeModel(fn_snapshot);
            manager.ShowModel();
            cout << "Model has shown successfully" << endl;
        }
    }
    else if (mode == 1) {
        if (!manager.LoadData(fn_list)) {
        cout << "Cannot load bounds" 
             << " Check " << fn_list << endl; 
        }
        else {
            cout << "Data loaded successfully." << endl;
        }
        if (!manager.LoadSilenceModel(fn_sil)) {
            cout << "Cannot load silence model " 
                 << "Check " << fn_sil << endl;
        }
        if (!config.LoadSeedingMixtures(fn_gseed)) {
            cout << "Cannot load gaussian seeds"
                << " Check " << fn_gseed << endl;
            exit(-1);
        }
        else {
            cout << "Loaded Gaussian Seeds successfully." << endl;
        }
        if (fn_snapshot == "") {
            manager.InitializeModel();
        }
        else {
            manager.InitializeModel(fn_snapshot);
        }
        manager.Inference(batch_size, n_iter, basedir);
    }
    else {
        cout << "Undefined mode: [0: read model; 1: training]" << endl;
    }
    cout << "Returning" << endl;
    return 0;
}

void usage() {
    cout << "gibbs -m [0: read model; 1: training] -l data_list -c configuration -n num_iteration " 
        << "-g gaussian_prior -s silence_model -b basedir -z batch_size -gs gaussian_seeds " 
        << "-snapshot snapshot_file" << endl;
}
