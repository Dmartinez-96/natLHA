#!/bin/bash
cd /mnt/c/Users/dakot/Documents/Research/DEW_code/softsusy-4.1.10
python softsusy_single_NUHM3_input_writer_5.py
./softpoint.x leshouches < test_in_5 > test_out_5
python softsusy_single_NUHM3_output_analyzer_cpu_5.py
