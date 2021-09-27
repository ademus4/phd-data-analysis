#include "GraphParameters.C"

void pshm_fit(string data_path, 
              string data_tree,
              string sim_data_path,
              string sim_data_tree,
	            string output_path,
              int bins,
              float bmin,
              float bmax,
              int nEvents,
	            bool mcmc,
              int L,
              int M,
              int nCores){
  FitManager Fitter;
  Fitter.SetUp().SetOutDir(output_path);

  Fitter.SetUp().LoadVariable("Pi2MesonGJCosTh[0,-1,1]");
  Fitter.SetUp().LoadVariable("Pi2MesonGJPhi[-3.14159,3.14159]");
  Fitter.SetUp().LoadVariable("Pi2MesonEPhi[-3.14159,3.14159]");
  Fitter.SetUp().LoadVariable("Pi2Pol[0.5,0.2,0.6]");
 

 // try L = 4 and 6 (8)
  auto configFitPDF=HS::FIT::EXPAND::ComponentsPolSphHarmonic(Fitter.SetUp(),
							      "Moments",
							      "Pi2MesonGJCosTh",
							      "Pi2MesonGJPhi",
							      "Pi2MesonEPhi",
							      "Pi2Pol",
							      L,M); //Moments is the refernce, add kTRUE for even waves only

  Fitter.SetUp().FactoryPDF(configFitPDF);
  Fitter.SetUp().LoadSpeciesPDF("Moments",nEvents); //2000 events
  Fitter.Bins().LoadBinVar("Pi2MesonMass",bins,bmin,bmax);  // nbins, min, max

  //Get the generated data
  Fitter.LoadData(data_tree, data_path);

  //flat data for mc integration
  Fitter.LoadSimulated(sim_data_tree, sim_data_path, "Moments");
  
  //run the fit
  gBenchmark->Start("fitting");
  if (mcmc){
    //RooMcmcSeq: n samples, burn in, steps (actually 1/step)
    Fitter.SetMinimiser(new RooMcmcSeqThenCov(40000,10000,100,10000,1000,1)); 
    Fitter.SetUp().AddFitOption(RooFit::Optimize(1));
  } else {
    Fitter.SetMinimiser(new Minuit2());
  }
  //Here::Go(&Fitter); 
  Proof::Go(&Fitter, nCores);
  gBenchmark->Stop("fitting");
  gBenchmark->Print("fitting");
}
