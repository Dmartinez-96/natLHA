#!/bin/bash
check_continue_4=true
while $check_continue_4; do if (( $(echo "$test_val_4 <= 30" | bc -l) ))
then echo "Found one!"
check_continue_4=false
else echo "DEW4 = $test_val_4";
test_val_4=$(./test_shell_script_4.sh)
fi
done
