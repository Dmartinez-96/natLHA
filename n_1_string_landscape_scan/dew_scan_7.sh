#!/bin/bash
check_continue_7=true
while $check_continue_7; do if (( $(echo "$test_val_7 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_7=false
else echo "DEW7 = $test_val_7";
test_val_7=$(./test_shell_script_7.sh)
fi
done
