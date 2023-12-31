//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class

#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "CaloValue.hh"

#include "G4Step.hh"
#include "G4AnalysisManager.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"

using namespace B4;

namespace B4a
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(const DetectorConstruction* detConstruction,
                               EventAction* eventAction)
  : fDetConstruction(detConstruction),
    fEventAction(eventAction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step)
{
// Collect energy and track length step by step

  // get volume of the current step
  auto volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();

  // energy deposit
  auto edep = step->GetTotalEnergyDeposit();
  // deposit position
  auto pos = step->GetPreStepPoint()->GetPosition();
  // step length
  G4double stepLength = 0.;
  if ( step->GetTrack()->GetDefinition()->GetPDGCharge() != 0. ) {
    stepLength = step->GetStepLength();
  }

  #ifdef STEP

  G4cout << "--- StepAction: edep "<< edep <<G4endl;
  G4cout << "--- StepAction: pos "<< pos <<G4endl;

  // get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->SetNtupleMerging(true);

  // fill ntuple
  analysisManager->FillNtupleDColumn(0, edep);
  analysisManager->FillNtupleDColumn(1, pos.x());
  analysisManager->FillNtupleDColumn(2, pos.y());

  analysisManager->AddNtupleRow();

  #endif

  if ( volume == fDetConstruction->GetGapPV() ) {
    fEventAction->AddGap(edep,stepLength);
    G4cout << "--- StepAction: edep in gap "<< edep <<G4endl;
  }
  else {
    for (int i = 0; i < NofCells * NofCells; i++) {
      if(volume == fDetConstruction->GetSensitivePV()[i].pv) {
        fEventAction->AddSen(i,edep, stepLength);
        G4cout << "--- StepAction: edep in sen "<<i << " is "<< edep/keV <<" kev"<<G4endl;
        return;
      }
      if ( volume == fDetConstruction->GetAbsorberPV()[i].pv ) {
        fEventAction->AddAbs(i, edep,stepLength);
        G4cout << "--- StepAction: edep in abs "<<i << " is "<< edep/keV <<" kev"<<G4endl;
        return;
      }
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}
