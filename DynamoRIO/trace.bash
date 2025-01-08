#!/bin/bash
rm -rf drmemtrace.*
rm *.trace

echo "random trace"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M  -- ./random_access
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.random_access* -compress lz4 -sim_refs 10M 2>&1 | tee random.trace

echo "graph computing"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- ./ligra/apps/PageRank -s ./ligra/inputs/rMat_1000000
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.PageRank* -compress lz4 -sim_refs 10M 2>&1 | tee logs/PageRank.trace

echo "HPC trace"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M  -- /home/zsq/DynamoRIO-Linux-10.93.19965/hpc-mpi-projects/distance-matrix/act1/distance_act1_mp2525.out
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.correlated_normal_dist* -compress lz4 -sim_refs 10M 2>&1 | tee HPC1.trace

echo "dnn trace"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- ./tiny-dnn/examples/benchmarks_all
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.benchmarks_all* -compress lz4 -sim_refs 10M 2>&1 | tee logs/DNN.trace

echo "ResNet trace"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- python ./PyTorch-Networks/ResNetTest.py
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- python ./PyTorch-Networks/YOLOv3Test.py
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.python* -compress lz4 -sim_refs 10M 2>&1 | tee logs/Net.trace

echo "HPC trace"
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- ./hpc-mpi-projects/distance-matrix/act1/distance_act1_mp2525.out 10000 90 ./hpc-mpi-projects/distance-matrix/MSD_year_prediction_normalize_0_1_100k.txt
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.distance_act1_mp2525* -compress lz4 -sim_refs 10M 2>&1 | tee logs/HPC.trace
./ligra/apps/PageRank -s ./ligra/inputs/rMatGraph_J_5_100

echo "MapReduce trace" && \
bin64/drrun -t drmemtrace -offline -L0I_filter -L0I_size 0 -skip_refs 10 -max_trace_size 10M -- ./MapReduce-Word-Count/host.o && \
bin64/drrun -t drmemtrace -tool view -indir drmemtrace.host.o* -compress lz4 -sim_refs 10M 2>&1 | tee logs/MapReduce.trace