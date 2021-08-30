#!/bin/bash
cd /mnt/c/Users/dakot/Documents/Research/DEW_code/softsusy-4.1.10
python softsusy_single_NUHM3_input_writer_2.py
./softpoint.x leshouches < test_in_2 > test_out_2
python softsusy_single_NUHM3_output_analyzer_cpu_2.py
