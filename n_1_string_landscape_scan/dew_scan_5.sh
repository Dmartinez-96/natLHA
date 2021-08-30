#!/bin/bash
check_continue_5=true
while $check_continue_5; do if (( $(echo "$test_val_5 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_5=false
else echo "DEW5 = $test_val_5";
test_val_5=$(./test_shell_script_5.sh)
fi
done
