int Run_Pi2(char * inputFilename, 
	    string config, 
	    string outputDir){

  // why dont env variables work here?
  clas12databases::SetCCDBLocalConnection("/home/adamt/dev/clas12root/RunRoot/ccdb.sqlite");
  clas12databases::SetRCDBRootConnection("/home/adamt/dev/clas12root/RunRoot/rcdb.root");
  
  ////Set hipo file to be analysed
  HipoData hdata;
  hdata.SetFile(inputFilename); //SetFile?
  hdata.LoadAnaDB("$CHANSER/rga_actions/anadb/RGA_ACTIONS_PASS1.db");
  hdata.LoadAnaDB("$CHANSER/anadbs/RunPeriodPass1.db");
  hdata.SetRunPeriod("fall_2018");
  hdata.Reader()->useFTBased();

  ////create FinalStateManager
  FinalStateManager fsm;
  fsm.SetBaseOutDir(outputDir); // 

  ////Connect the data to the manager
  fsm.LoadData(&hdata);
  
  ////load one or more FinalStates
  fsm.LoadFinalState("Pi2", config); //

  //Max number of particles of any 1 species
  //Whole event disgarded if this not met.
  fsm.GetEventParticles().SetMaxParticles(6);

  ////Run through all events
  fsm.ProcessAll();
  return 0;
}


//TEnvironment, could be used for systematics, ask Peter
