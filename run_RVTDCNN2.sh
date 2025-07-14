#!/bin/bash

# Set the Python file name
#python_file="DPDtransformer.py"
 python_file="RVTDCNN2.py"

# Set the variable name to monitor
variable_name="NMSE"
sum=0
count=10

# Print the test file name
echo "Test on $python_file"

# Loop through 10 runs of the Python file
for ((i=1;i<=$count;i++));
do
  # Run the Python file and save the output to a variable
  output=$(CUDA_VISIBLE_DEVICES=4 python $python_file $i)

  # Extract the value of the monitored variable from the output
  variable_value=$(echo "$output" | grep "$variable_name" | awk '{print $NF}')

  # Print the value of the variable to the console
  echo "Run $i: $variable_value"

  # 将值添加到总和中
  sum=$(echo "$sum + $variable_value" | bc)
  # Wait for 1 second before starting the next run
  sleep 1
done

# 计算平均值
average=$(echo "scale=2; $sum / $count" | bc)

# Extract the value of the num of parameters from the output
param_num=$(echo "$output" | grep "parameters" | awk '{print $NF}')

# Extract the wandb configuration from the output
wandb_configuration=$(echo "$output" | grep "wandb config" | awk '{print $0}')

echo "wandb configuration: $wandb_configuration"
echo "Total number of parameters: $param_num"
echo "The average NMSE is: $average"