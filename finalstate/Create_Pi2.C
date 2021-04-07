{
  auto FS = adamt::Pi2::Make("NONE","ALL"); // removal second all, exact number (exclusive)
  FS->AddTopology("Electron:Proton:Pip:Pim");
  FS->AddTopology("Electron:Proton:Pim");
  FS->AddTopology("Electron:Proton:Pip");
  FS->AddTopology("Electron:Pip:Pim");

  //Save TreeDataPi2
  FS->UseOutputRootTree();

  /////Make particle trees first in case want to add cut flags
  ParticleDataManager pdm{"particle",1};
  pdm.SetParticleOut(new CLAS12ParticleOutEvent0);
  FS->RegisterPostKinAction(pdm);

  //apply some general cuts
  ParticleCutsManager pcm{"DeltaTimeCuts",1};  //1==apply!
  pcm.AddParticleCut("e-",    new DeltaTimeCut(10));
  pcm.AddParticleCut("proton",new DeltaTimeCut(10));
  pcm.AddParticleCut("pi+",   new DeltaTimeCut(10));
  pcm.AddParticleCut("pi-",   new DeltaTimeCut(10));
  FS->RegisterPostTopoAction(pcm);

  //for simulations, to correct for start time
  //FS->SetStartTimePeak(124.25);
  //FS->HalveBunchTime();

  //set start time
  //StartTimeAction st("StartTime",new C12StartTimeFromParticle("Electron"));
  StartTimeAction st("EBStartTime",new C12StartTimeFromVtFTB());
  FS->RegisterPreTopoAction(st);

  //FT electron energy correction
  ParticleCorrectionManager pcorrm{"FTelEnergyCorrection"};
  pcorrm.AddParticle("e-",new FTel_pol4_ECorrection());
  FS->RegisterPreTopoAction(pcorrm);

  //write out config
  FS->WriteToFile("./finalstate/config.root");
  FS->Print();

  //Delete the final state rather than let ROOT try
  FS.reset();

  // need to re-run
  // chanser_root Pi2:Pi2.cpp Create_Pi2.C
}
