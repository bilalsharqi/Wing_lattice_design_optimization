Lattice Optimization V4 (hard connectivity constraints)
=======================================================

Key changes
-----------
1. Intact graph must remain connected after every pruning step.
2. Damaged graph must remain connected; otherwise design fails.
3. Tip displacement is measured on the connected load-carrying graph.
4. Near-mechanism behavior is flagged using a stiffness condition estimate.
5. HDF5 stores connectivity flags, tip node indices, and condition estimates.

Run
---
pip install numpy scipy matplotlib networkx h5py
python optimize_v4_connected.py