{
  auto FS = adamt::Pi2::Make("NONE","ALL"); // (use EB PID, exact number, exclusive)
  FS->AddTopology("Electron:Proton:Pip:Pim");
  FS->AddTopology("Electron:Proton:Pim");
  FS->AddTopology("Electron:Proton:Pip");
  FS->AddTopology("Electron:Pip:Pim");

  //Save TreeDataPi2
  FS->UseOutputRootTree();
  
  //apply some general cuts
  ParticleCutsManager pcm{"DeltaTimeCuts",1};  //1==apply!
  pcm.AddParticleCut("e-",    new DeltaTimeVerCut(1));
  pcm.AddParticleCut("proton",new DeltaTimeVerCut(1));
  pcm.AddParticleCut("pi+",   new DeltaTimeVerCut(1));
  pcm.AddParticleCut("pi-",   new DeltaTimeVerCut(1));
  FS->RegisterPostTopoAction(pcm);

  //0.5ns cut
  ParticleCutsManager pcm2{"DeltaTimeCuts_05",0};
  pcm2.AddParticleCut("e-",    new DeltaTimeVerCut(0.5));
  pcm2.AddParticleCut("proton",new DeltaTimeVerCut(0.5));
  pcm2.AddParticleCut("pi+",   new DeltaTimeVerCut(0.5));
  pcm2.AddParticleCut("pi-",   new DeltaTimeVerCut(0.5));
  FS->RegisterPostTopoAction(pcm2);

  //0.2ns cut
  ParticleCutsManager pcm3{"DeltaTimeCuts_02",0};
  pcm3.AddParticleCut("e-",    new DeltaTimeVerCut(0.2));
  pcm3.AddParticleCut("proton",new DeltaTimeVerCut(0.2));
  pcm3.AddParticleCut("pi+",   new DeltaTimeVerCut(0.2));
  pcm3.AddParticleCut("pi-",   new DeltaTimeVerCut(0.2));
  FS->RegisterPostTopoAction(pcm3);

  //correction for electron
  ParticleCorrectionManager pVz{"FTelVz"};//1=> for simulation too
  pVz.AddParticle("e-",new FTel_VzCorrection(-0.05));//5cm shift
  FS->RegisterPreTopoAction(pVz);

  //set start time
  StartTimeAction st("StartTime",new C12StartTimeFromParticle("Electron"));  //better
  FS->RegisterPreTopoAction(st);

  //FT electron energy correction
  ParticleCorrectionManager pcorrm{"FTelEnergyCorrection"};
  pcorrm.AddParticle("e-",new FTel_pol4_ECorrection());
  FS->RegisterPreTopoAction(pcorrm);

  //truth matching
  EventTruthAction etra("EventTruth");
  FS->RegisterPostKinAction(etra); //PostKin

  //write out config
  FS->WriteToFile("./finalstate/config.root");
  FS->Print();

  //Delete the final state rather than let ROOT try
  FS.reset();

  // need to re-run
  // chanser_root Pi2:Pi2.cpp Create_Pi2.C
}
