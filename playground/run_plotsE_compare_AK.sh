#!/bin/bash

histograms_folders=(
  Histograms_ECM240_AK_organized_NoISR
  Histograms_ECM240_AK_organized_ISR
)

folder_names=(
  p8_ee_ZH_6jet_ecm240
  p8_ee_ZH_bbbb_ecm240
  p8_ee_ZH_qqbb_ecm240
  p8_ee_ZH_vvbb_ecm240
  p8_ee_ZH_vvgg_ecm240
)

for hist in "${histograms_folders[@]}"; do
  for folder in "${folder_names[@]}"; do
    echo "▶ Running with HISTOGRAMS_FOLDER_NAME=$hist, FOLDER_NAME=$folder"
    HISTOGRAMS_FOLDER_NAME="$hist" FOLDER_NAME="$folder" \
      fccanalysis plots plots_jetE_alljets_Compare_AK.py
  done
done
