#!/bin/bash
check_continue_8=true
while $check_continue_8; do if (( $(echo "$test_val_8 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_8=false
else echo "DEW8 = $test_val_8";
test_val_8=$(./test_shell_script_8.sh)
fi
done
