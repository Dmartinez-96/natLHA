#!/bin/bash
check_continue_1=true
while $check_continue_1; do if (( $(echo "$test_val <= 30" | bc -l) ))
then echo "Found one!"
check_continue_1=false
else echo "DEW = $test_val";
test_val=$(./test_shell_script.sh)
fi
done
