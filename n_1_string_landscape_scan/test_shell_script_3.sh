#!/bin/bash
cd /mnt/c/Users/dakot/Documents/Research/DEW_code/softsusy-4.1.10
python softsusy_single_NUHM3_input_writer_3.py
./softpoint.x leshouches < test_in_3 > test_out_3
python softsusy_single_NUHM3_output_analyzer_cpu_3.py
