#!/bin/bash
cd /mnt/c/Users/dakot/Documents/Research/DEW_code/softsusy-4.1.10
python softsusy_single_NUHM3_input_writer.py
./softpoint.x leshouches < test_in > test_out
python softsusy_single_NUHM3_output_analyzer_cpu.py
