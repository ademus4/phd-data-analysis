#pragma once


#include "BaseOutEvent.h"

#pragma link C++ class adamt::TreeDataPi2;

namespace adamt{
  
  class TreeDataPi2 : public chanser::BaseOutEvent{
    
  public:
    TreeDataPi2(){SetName("Pi2");}
    ~TreeDataPi2() final =default;
      
    //data member for tree branches below here
    Double_t MissMass2=0;
    Double_t MissMass=0;
    Double_t MissE=0;
    Double_t MissP=0;
    
    Double_t MissMass2nP=0;
    Double_t MissMassnP=0;
    Double_t MissEnP=0;
    Double_t MissPnP=0;

    Double_t MissMass2nPim=0;
    Double_t MissMassnPim=0;
    Double_t MissEnPim=0;
    Double_t MissPnPim=0;

    Double_t MissMass2nPip=0;
    Double_t MissMassnPip=0;
    Double_t MissEnPip=0;
    Double_t MissPnPip=0;
    
    //trigger related
    Double_t TriggerElectron=0;
    Double_t TriggerMesonex=0;
    Double_t Trigger24=0;
    Double_t TriggerFT2C=0;
    Double_t TriggerFT100=0;
    Double_t TriggerSim=0;
    Double_t TriggerFDSectors=0;
    Double_t TriggerTotalEnergyFT=0;
    Double_t PCALEnergy=0;
    Double_t FTOF1BEnergy=0;
    Double_t DCNDF0=0;

    Double_t MesonMass=0;
    Double_t DppMass=0;
    Double_t D0Mass=0;
  
    Double_t W=0; 
    Double_t Q2=0;
    Double_t Pol=0;
    Double_t Egamma=0;

    //mom transferred
    Double_t t=0;
  
    //meson angles in various reference frames
    Double_t MesonECosTh=0;
    Double_t MesonEPhi=0;  
    Double_t MesonHCosTh=0;
    Double_t MesonHPhi=0;  
    Double_t MesonGJCosTh=0;
    Double_t MesonGJPhi=0;    
  
    Double_t ElTh=0;
    Double_t ElP=0;
    Double_t ElTime=0;
    Double_t ElDE=0;
    Double_t ElRegion=0;
    Double_t ElBeta=0;
    Double_t ElBeta2=0;

    Double_t ProtTh=0;
    Double_t ProtP=0;
    Double_t ProtTime=0;
    Double_t ProtRegion=0;
    Double_t ProtBeta=0;
    Double_t ProtBeta2=0;

    Double_t PipTh=0;
    Double_t PipP=0;
    Double_t PipTime=0;
    Double_t PipRegion=0;
    Double_t PipBeta=0;
    Double_t PipBeta2=0;

    Double_t PimTh=0;
    Double_t PimP=0;
    Double_t PimTime=0; 
    Double_t PimRegion=0;
    Double_t PimBeta=0;
    Double_t PimBeta2=0;

    ///////////////////////////////////////////////////////////
    //LEAVE THE FOLLOWING FUNCTIONS
    //Function required to set tree branches
    void Branches(TTree* tree) final{
      BaseOutEvent::Branches(tree,Class()->GetListOfDataMembers());
    }
    void Hipo(hipo::ntuple_writer* writer) final{
      BaseOutEvent::Hipo(writer,Class()->GetListOfDataMembers());
    }
      
    ClassDefOverride(TreeDataPi2,1);
  };
}
