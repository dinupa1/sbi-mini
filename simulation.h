#include <TFile.h>
#include <TTree.h>
#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TSystem.h>
#include <iostream>

int num_samples = 100000;
double pi = TMath::Pi();
double lumi_up = 0.5;
double pol_up = 0.8;
double pol_down = 0.8;
double AN_data = 0.1;
double AN_sim = 0.0;
double effi_sim = 0.0;
double effi_data = 0.5;
double smear = 0.5;
double two_pi = 2.* pi;

double T_sim;
double R_sim;
double pol_sim;
double phi_pol_sim;
double weight_sim;
double zeros;

double T_data;
double R_data;
double pol_data;
double phi_pol_data;
double weight_data;
double ones;


TTree* tree_sim;
TTree* tree_data;
