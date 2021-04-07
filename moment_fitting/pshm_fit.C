#include "GraphParameters.C"

void pshm_fit(string data_path, 
              string data_tree,
              string sim_data_path,
              string sim_data_tree,
	            string output_path,
              int nEvents,
	            bool mcmc){
  FitManager Fitter;
  Fitter.SetUp().SetOutDir(output_path);

  Fitter.SetUp().LoadVariable("Pi2MesonGJCosTh[0,-1,1]");
  Fitter.SetUp().LoadVariable("Pi2MesonGJPhi[-3.14159,3.14159]");
  Fitter.SetUp().LoadVariable("Pi2MesonEPhi[-3.14159,3.14159]");
  Fitter.SetUp().LoadVariable("Pi2Pol[0.5,0.2,0.6]");
  Fitter.SetUp().LoadVariable("Pi2MesonMass[0,3]");
 
  auto configFitPDF=HS::FIT::EXPAND::ComponentsPolSphHarmonic(Fitter.SetUp(),
							      "Moments",
							      "Pi2MesonGJCosTh",
							      "Pi2MesonGJPhi",
							      "Pi2MesonEPhi",
							      "Pi2Pol",
							      3,2); //Moments is the refernce, add kTRUE for even waves only

  //cout<<"##########"<<endl;
  //cout<<configFitPDF<<endl;
  //return

  Fitter.SetUp().FactoryPDF(configFitPDF);
  Fitter.SetUp().LoadSpeciesPDF("Moments",nEvents); //2000 events
  Fitter.Bins().LoadBinVar("Pi2MesonMass",3,0,3);  // nbins, min, max

  //Get the generated data
  Fitter.LoadData(data_tree, data_path);

  //flat data for mc integration
  Fitter.LoadSimulated(sim_data_tree, sim_data_path, "Moments");
  
  //run the fit
  gBenchmark->Start("fitting");
  if (mcmc){
    Fitter.SetMinimiser(new RooMcmcSeq(2000,1000,50));
    Fitter.SetUp().AddFitOption(RooFit::Optimize(1));
  } else {
    Fitter.SetMinimiser(new Minuit2());
  }
  Here::Go(&Fitter); 
  gBenchmark->Stop("fitting");
  gBenchmark->Print("fitting");

  //merge the meson mass bins
  GraphParameters(output_path,"Pi2MesonMass");  // not working?
}
