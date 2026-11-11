#!/bin/bash
# 11-09 RS-group decomposition batch (planner stage 2, RS part). One setsid chain, logs under results/raw/e2e_8card/rs1109_*.
#   P1 70B MBS2 (D~77, the only measured RS-group loss): base / fnull / fdk / frow0 / frowk, ABAB
#   P2 70B MBS4 (D~134, RS group wins): base / fdk / frow0, ABAB
#   P3 27B MBS4 (D~74, RS group never measured): base / fdk / frow0, ABAB
set -u
HERE=$(dirname "$(readlink -f "$0")")
R=$HERE/../../results/raw/e2e_8card
echo "[chain] start $(date +%F' '%T)"
SW_MODEL=70b MBS=2 ITERS=14 E2E_PORT=26000 SW_OUT=$R/rs1109_70b_mbs2 \
  ARMS="b1:base n1:fnull d1:fdk r1:frow0 k1:frowk b2:base n2:fnull d2:fdk r2:frow0 k2:frowk" bash $HERE/run.sh
echo "[chain] P1 done $(date +%T)"
SW_MODEL=70b MBS=4 ITERS=14 E2E_PORT=27000 SW_OUT=$R/rs1109_70b_mbs4 \
  ARMS="b1:base d1:fdk r1:frow0 b2:base d2:fdk r2:frow0" bash $HERE/run.sh
echo "[chain] P2 done $(date +%T)"
SW_MODEL=27b MBS=4 ITERS=14 E2E_PORT=28000 SW_OUT=$R/rs1109_27b_mbs4 \
  ARMS="b1:base d1:fdk r1:frow0 b2:base d2:fdk r2:frow0" bash $HERE/run.sh
echo "[chain] P3 done $(date +%T)"
echo "[chain] ALL DONE $(date +%F' '%T)"
