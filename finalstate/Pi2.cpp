  
#include "Pi2.h"
 
namespace adamt{
  
  ///////////////////////$$$$$$$$$$$$$$$$$$$$$$$$$$//////////////////////  
  void Pi2::Define(){
    //Set final state detected particles
    //Note if particle is added to final with a valid genID it will be used
    //to determine the correct permutation of the simulated event
      
    //last value for MC truth matching, id from el spectro 2pi script
    AddParticle("Electron",&_electron,kTRUE, 3);  //3
    AddParticle("Proton",&_proton,kTRUE, 2);      //2
    AddParticle("Pip",&_pip,kTRUE, 0);            //0
    AddParticle("Pim",&_pim,kTRUE, 1);            //1
    //AddParticle("Name",particle,true/false you want to write in final vector, genID for linking to generated truth value)
    // AddParticle("PARTICLE",&_PARTICLE,kTRUE,-1);

    //Set Possible Topologies
    _doToTopo["Electron:Proton:Pip:Pim"]=[&](){
      //TOPOLOGY Define your topology dedendent code in here
      ///////+++++++++++++++++++++++++++++++++++///////

      //only events with FT electrons
      if(_electron.CLAS12()->getRegion()!=1000){RejectEvent(); return;}

      //calc for the diff combinations (missing parts)
      auto miss= _beam + _target - _electron.P4() - _proton.P4() - _pip.P4() - _pim.P4();
      TD->MissMass2=miss.M2();    
      TD->MissMass =miss.M();
      TD->MissE    =miss.E();
      TD->MissP    =miss.P();

      auto missp= _beam + _target - _electron.P4() - _pip.P4() - _pim.P4();
      TD->MissMass2nP =missp.M2();
      TD->MissMassnP  =missp.M();
      TD->MissEnP     =missp.E();
      TD->MissPnP     =missp.P();

      auto misspim= _beam + _target - _electron.P4() - _proton.P4() - _pip.P4();
      TD->MissMass2nPim =misspim.M2();
      TD->MissMassnPim  =misspim.M();
      TD->MissEnPim     =misspim.E();
      TD->MissPnPim     =misspim.P();

      auto misspip= _beam + _target - _electron.P4() - _proton.P4() - _pim.P4();
      TD->MissMass2nPip =misspip.M2();
      TD->MissMassnPip  =misspip.M();
      TD->MissEnPip     =misspip.E();
      TD->MissPnPip     =misspip.P();

      TD->ElTh       =_electron.CLAS12()->getTheta();
      TD->ElP        =_electron.CLAS12()->getP();
      TD->ElTime     =_electron.CLAS12()->getTime();
      TD->ElDE       =_electron.CLAS12()->getDeltaEnergy();
      TD->ElRegion   =_electron.CLAS12()->getRegion();
      TD->ElBeta     =_electron.Beta();
      TD->ElBeta2    =_electron.BetaVer();

      TD->ProtTh     =_proton.CLAS12()->getTheta();
      TD->ProtP      =_proton.CLAS12()->getP();
      TD->ProtTime   =_proton.CLAS12()->getTime();
      TD->ProtRegion =_proton.CLAS12()->getRegion();
      TD->ProtBeta   =_proton.Beta();
      TD->ProtBeta2  =_proton.BetaVer();

      TD->PipTh      =_pip.CLAS12()->getTheta();
      TD->PipP       =_pip.CLAS12()->getP();
      TD->PipTime    =_pip.CLAS12()->getTime();
      TD->PipRegion  =_pip.CLAS12()->getRegion();
      TD->PipBeta    =_pip.Beta();
      TD->PipBeta2   =_pip.BetaVer();

      TD->PimTh      =_pim.CLAS12()->getTheta();
      TD->PimP       =_pim.CLAS12()->getP();
      TD->PimTime    =_pim.CLAS12()->getTime();
      TD->PimRegion  =_pim.CLAS12()->getRegion();
      TD->PimBeta    =_pim.Beta();
      TD->PimBeta2   =_pim.BetaVer();

      ///////------------------------------------///////
    };
    _doToTopo["Electron:Proton:Pip"]=[&](){
      //TOPOLOGY Define your topology dedendent code in here
      ///////+++++++++++++++++++++++++++++++++++///////

      if(_electron.CLAS12()->getRegion()!=1000){RejectEvent(); return;}
      
      auto miss= _beam + _target - _electron.P4() - _proton.P4() - _pip.P4();
      TD->MissMass2=miss.M2();    
      TD->MissMass =miss.M();
      TD->MissE=miss.E();
      _pim.FixP4(miss);
      ///////------------------------------------///////
    };
    _doToTopo["Electron:Proton:Pim"]=[&](){
      //TOPOLOGY Define your topology dedendent code in here
      ///////+++++++++++++++++++++++++++++++++++///////
      if(_electron.CLAS12()->getRegion()!=1000){RejectEvent(); return;}

      auto miss= _beam + _target - _electron.P4() - _proton.P4() - _pim.P4();
      TD->MissMass2=miss.M2();    
      TD->MissMass =miss.M();
      TD->MissE=miss.E();
      _pip.FixP4(miss);
      ///////------------------------------------///////
    };
    _doToTopo["Electron:Pip:Pim"]=[&](){
      //TOPOLOGY Define your topology dedendent code in here
      ///////+++++++++++++++++++++++++++++++++++///////
      if(_electron.CLAS12()->getRegion()!=1000){RejectEvent(); return;}

      auto miss= _beam + _target - _electron.P4() - _pip.P4() - _pim.P4();
      TD->MissMass2=miss.M2();    
      TD->MissMass =miss.M();
      TD->MissE=miss.E();
      _proton.FixP4(miss);
      ///////------------------------------------///////
    };
      
  }

 
  ///////////////////////$$$$$$$$$$$$$$$$$$$$$$$$$$//////////////////////  
  void Pi2::Kinematics(){
    //Define reaction specific kinematic calculations here
    //Assign to tree data TD.var=
    
    //Use Kinematics to calculate electron variables
    //Note this assumes you called your electron "electron" or "Electron"
    _kinCalc.SetElecsTarget(_beam,_electron.P4(),_target);
    TD->W=_kinCalc.W(); //photon bem energy
    TD->Q2=_kinCalc.Q2();
    TD->Pol=_kinCalc.GammaPol();
    TD->Egamma=_kinCalc.GammaE();
    
    //Calculate possible resonances
    auto pMeson=_pip.P4()+_pim.P4();
    auto pDpp=_proton.P4()+_pip.P4();
    auto pD0=_proton.P4()+_pim.P4();
    
    //mom transferred alt
    //auto pGamma=_kinCalc.Gamma();
    //TD->t=(pGamma-pMeson).M2();

    //trigger related
    if (IsGenerated()==false){
      TD->TriggerMesonex=GetEventInfo()->CLAS12()->checkTriggerBit(25);
    }
    
    //invariant masses
    TD->MesonMass=pMeson.M();
    TD->DppMass=pDpp.M();
    TD->D0Mass=pD0.M();
     
    //calculate CM production kinematics for meson
    _kinCalc.SetMesonBaryon(pMeson,_proton.P4());
    _kinCalc.ElectroCMDecay();
    TD->MesonECosTh=_kinCalc.CosTheta();//prefix all variables to be saved wiht TM.
    TD->MesonEPhi=_kinCalc.Phi();
    TD->t=_kinCalc.t();

    //helicity
    _kinCalc.SetMesonDecay(_pip.P4(),_pim.P4());
    _kinCalc.MesonDecayHelicity();
    TD->MesonHCosTh=_kinCalc.CosTheta();
    TD->MesonHPhi=_kinCalc.Phi();
    
    //GJ
    _kinCalc.SetMesonDecay(_pip.P4(),_pim.P4());
    _kinCalc.MesonDecayGJ();
    TD->MesonGJCosTh=_kinCalc.CosTheta();
    TD->MesonGJPhi=_kinCalc.Phi();
  }
    
  ///////////////////////$$$$$$$$$$$$$$$$$$$$$$$$$$//////////////////////  
  void Pi2::UserProcess(){
    //Optional additional steps
    //This is called after the Topo function
    //and before the kinematics function
    _counter++;


    //Must call the following to fill Final trees etc.
    //Fill Final data output at the end
    FinalState::UserProcess();

  }
  
  
}
