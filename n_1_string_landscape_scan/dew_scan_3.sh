#!/bin/bash
check_continue_3=true
while $check_continue_3; do if (( $(echo "$test_val_3 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_3=false
else echo "DEW3 = $test_val_3";
test_val_3=$(./test_shell_script_3.sh)
fi
done
