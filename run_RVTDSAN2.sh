#!/bin/bash

# Set the Python file name
python_file="RVTDSAN2.py"

# Set the variable name to monitor
variable_name="NMSE"
sum=0
count=5
neuronsN=0
# Print the test file name
echo "Test on $python_file"

for ((j=1;j<=4;j++)); do
  # Loop through 10 runs of the Python file
  sum=0
  let "neuronsN = j + 8"
  for ((i=1;i<=$count;i++)); do
    # Run the Python file and save the output to a variable
#     if [ $j -eq 1 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'Sigmoid' )
#     fi
#     if [ $j -eq 2 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'Tanh')
#     fi
#     if [ $j -eq 3 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'ReLU')
#     fi
#     if [ $j -eq 4 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'Leaky ReLU')
#     fi
#     if [ $j -eq 5 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'GELU')
#     fi
#     if [ $j -eq 6 ]; then
#       output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i 'Swish')
#     fi

    output=$(CUDA_VISIBLE_DEVICES=3 python $python_file $i 'Leaky ReLU' $neuronsN)

    # Extract the value of the monitored variable from the output
    variable_value=$(echo "$output" | grep "$variable_name" | awk '{print $NF}')

    # Print the value of the variable to the console
    echo "Run $i: $variable_value"

    # Add the value to the sum
    sum=$(echo "$sum + $variable_value" | bc)

    # Wait for 1 second before starting the next run
#     sleep 1
  done

  # Calculate the average value
  average=$(echo "scale=2; $sum / $count" | bc)

  # Extract the value of the num of parameters from the output
  param_num=$(echo "$output" | grep "parameters" | awk '{print $NF}')

#   # Extract the wandb configuration from the output
#   wandb_configuration=$(echo "$output" | grep "wandb config" | awk '{print $0}')

  # Extract the value of the num of parameters from the output
  param_num=$(echo "$output" | grep "macs" | awk '{print $0}')

  # Extract the value of the num of parameters from the output
  neurons_num=$(echo "$output" | grep "neurons" | awk '{print $0}')

  # Extract the activate function from the output
  activate_function=$(echo "$output" | grep "activate function" | awk '{print $0}')

  #   echo "$wandb_configuration"
  echo "$activate_function"
#   echo "$neurons_num"
  echo "Total number of parameters: $param_num"
  echo "The average NMSE is: $average"


done