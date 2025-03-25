#include "simulation.h"


int unbinned_unfolding() {

    TFile* outfile = new TFile("outfile.root", "recreate");

    tree_sim = new TTree("tree_sim", "tree_sim");

    tree_sim->Branch("T_sim",           &T_sim,         "T_sim/D");
    tree_sim->Branch("R_sim",           &R_sim,         "R_sim/D");
    tree_sim->Branch("pol_sim",         &pol_sim,       "pol_sim/D");
    tree_sim->Branch("phi_pol_sim",     &phi_pol_sim,   "phi_pol_sim/D");
    tree_sim->Branch("AN_sim",          &AN_sim,        "AN_sim/D");
    tree_sim->Branch("effi_sim",        &effi_sim,      "effi_sim/D");
    tree_sim->Branch("weight_sim",      &weight_sim,    "weight_sim/D");
    tree_sim->Branch("zeros",           &zeros,         "zeros/D");

    tree_data = new TTree("tree_data", "tree_data");

    tree_data->Branch("T_data",         &T_data,            "T_data/D");
    tree_data->Branch("R_data",         &R_data,            "R_data/D");
    tree_data->Branch("pol_data",       &pol_data,          "pol_data/D");
    tree_data->Branch("phi_pol_data",   &phi_pol_data,      "phi_pol_data/D");
    tree_data->Branch("AN_data",        &AN_data,           "AN_data/D");
    tree_data->Branch("effi_data",      &effi_data,         "effi_data/D");
    tree_data->Branch("weight_data",    &weight_data,       "weight_data/D");
    tree_data->Branch("ones",           &ones,              "ones/D");

    TRandom3* events = new TRandom3(42);

    //
    // simulation
    //
    for(int ii = 0; ii < num_samples; ii++) {

        T_sim = events->Uniform(-pi, pi);

        double lumi = events->Uniform(0., 1.);

        if(lumi > lumi_up) { //spin down
            pol_sim = pol_down;
            phi_pol_sim = -pi/2.;

            double weight = (1./two_pi)* (1. + pol_sim* AN_sim* sin(phi_pol_sim - T_sim));
            double weight_max = (1./two_pi)* (1. + pol_sim* AN_sim);
            double efficiency = 1;
            double threshold = events->Uniform(0., 1.);
            // if(threshold > (weight/weight_max)* efficiency){ii--; continue;}
            if(2.* threshold > weight* efficiency){ii--; continue;}

            double phi = events->Gaus(T_sim, smear);
            if(phi < -pi) {R_sim = phi + 2.0* pi;}
            else if(phi > pi) {R_sim = phi - 2.0* pi;}
            else {R_sim = phi;}

            weight_sim = (lumi_up* pol_up + (1. - lumi_up)* pol_down)/(2.* (1. - lumi_up)* pol_down);
        }

        else { // spin up
            pol_sim = pol_up;
            phi_pol_sim = pi/2.;

            double weight = (1./two_pi)* (1. + pol_sim* AN_sim* sin(phi_pol_sim - T_sim));
            double weight_max = (1./two_pi)* (1. + pol_sim* AN_sim);
            double efficiency = 1;
            double threshold = events->Uniform(0., 1.);
            // if(threshold > (weight/weight_max)* efficiency){ii--; continue;}
            if(2.* threshold > weight* efficiency){ii--; continue;}

            double phi = events->Gaus(T_sim, smear);
            if(phi < -pi) {R_sim = phi + 2.0* pi;}
            else if(phi > pi) {R_sim = phi - 2.0* pi;}
            else {R_sim = phi;}

            weight_sim = (lumi_up* pol_up + (1. - lumi_up)* pol_down)/(2.* lumi_up* pol_up);
        }

        zeros = 0.;

        tree_sim->Fill();
    }

    //
    // data
    //
    for(int ii = 0; ii < num_samples; ii++) {

        T_data = events->Uniform(-pi, pi);

        double lumi = events->Uniform(0., 1.);

        if(lumi > lumi_up) { //spin down
            pol_data = pol_down;
            phi_pol_data = -pi/2.;

            double weight = (1./two_pi)* (1. + pol_data* AN_data* sin(phi_pol_data - T_data));
            double weight_max = (1./two_pi)* (1. + pol_data* AN_data);
            double efficiency = 1.; // 0.5*(1. + effi_data* cos(T_data));
            double effi_max = 1. + effi_data;
            double threshold = events->Uniform(0., 1.);
            // if(threshold > (weight/weight_max)* (efficiency/effi_max)){ii--; continue;}
            if(2.* threshold > weight* efficiency){ii--; continue;}

            double phi = events->Gaus(T_data, smear);
            if(phi < -pi) {R_data = phi + 2.0* pi;}
            else if(phi > pi) {R_data = phi - 2.0* pi;}
            else {R_data = phi;}

            weight_data = (lumi_up* pol_up + (1. - lumi_up)* pol_down)/(2.* (1. - lumi_up)* pol_down);
        }

        else { // spin up
            pol_data = pol_up;
            phi_pol_data = pi/2.;

            double weight = (1./two_pi)* (1. + pol_data* AN_data* sin(phi_pol_data - T_data));
            double weight_max = (1./two_pi)* (1. + pol_data* AN_data);
            double efficiency = 1.; //0.5*(1. + effi_data* cos(T_data));
            double effi_max = 1. + effi_data;
            double threshold = events->Uniform(0., 1.);
            // if(threshold > (weight/weight_max)* (efficiency/effi_max)){ii--; continue;}
            if(2.* threshold > weight* efficiency){ii--; continue;}

            double phi = events->Gaus(T_data, smear);
            if(phi < -pi) {R_data = phi + 2.0* pi;}
            else if(phi > pi) {R_data = phi - 2.0* pi;}
            else {R_data = phi;}

            weight_data = (lumi_up* pol_up + (1. - lumi_up)* pol_down)/(2.* lumi_up* pol_up);
        }

        ones = 1.;

        tree_data->Fill();
    }

    tree_sim->Write();
    tree_data->Write();
    outfile->Close();

    //
    // unfolding
    //
    gSystem->Exec("python smearing_correction.py");

    return 0;
}
