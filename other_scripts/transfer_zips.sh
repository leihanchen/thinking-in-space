#!/bin/bash

# prerequisites:
# 1. pip environment with globus-cli installed
# 2. login to globus via `globus login` (you will need to verify via browser code and login); expires after 10 minutes

FIR_ENDPOINT=$(globus endpoint search alliancecan#fir | grep computecanada#cedar-globus | head -n 1 | cut -d ' ' -f 1)
NARVAL_ENDPOINT=$(globus endpoint search Narval | grep ComputeCanada | head -n 1 | cut -d ' ' -f 1)
NIBI_ENDPOINT=$(globus endpoint search alliancecan#nibi | grep alliancecan#nibi | head -n 1 | cut -d ' ' -f 1)
RORQUAL_ENDPOINT=$(globus endpoint search alliancecan#rorqual | grep alliancecan#rorqual | head -n 1 | cut -d ' ' -f 1)
TRILLIUM_ENDPOINT=$(globus endpoint search alliancecan#trillium | grep alliancecan#trillium | head -n 1 | cut -d ' ' -f 1)
TAMIA_ENDPOINT=$(globus endpoint search tamia | grep TamIA | head -n 1 | cut -d ' ' -f 1)


# globus transfer --recursive --verbose --include ".zip" ${TRILLIUM_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/ ${FIR_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/
# globus transfer --recursive --verbose --include ".zip" ${TRILLIUM_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/ ${NARVAL_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/
# globus transfer --recursive --verbose --include ".zip" ${TRILLIUM_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/ ${NIBI_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/
# globus transfer --recursive --verbose --include ".zip" ${TRILLIUM_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/ ${RORQUAL_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/
globus transfer --recursive --verbose --include '43*' --exclude '*.zip' ${TRILLIUM_ENDPOINT}:/scratch/indrisch/ARKitScenes_data/ ${TAMIA_ENDPOINT}:/scratch/i/indrisch/ARKitScenes_data/