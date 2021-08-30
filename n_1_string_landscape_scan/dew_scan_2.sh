#!/bin/bash
check_continue_2=true
while $check_continue_2; do if (( $(echo "$test_val_2 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_2=false
else echo "DEW2 = $test_val_2";
test_val_2=$(./test_shell_script_2.sh)
fi
done
