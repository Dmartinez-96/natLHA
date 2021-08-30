#!/bin/bash
check_continue_6=true
while $check_continue_6; do if (( $(echo "$test_val_6 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_6=false
else echo "DEW6 = $test_val_6";
test_val_6=$(./test_shell_script_6.sh)
fi
done
