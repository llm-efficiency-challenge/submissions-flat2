#!/bin/bash

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y bc

# Function to check CPU utilization
function check_cpu_utilization() {
  # Get the CPU utilization percentage using `top` command and extract the value using `grep` and `awk`
  local cpu_utilization=$(top -bn 1 | grep '%Cpu' | awk '{print $2}')
  echo "$cpu_utilization"
}

count=0
wait=60

echo "CPU monitor script started. Timeout set to $wait minutes."

# Main loop to monitor CPU utilization and shutdown the VM if idle
while true; do
  # Sleep for 60 seconds
  sleep 60

  # Get the current CPU utilization percentage
  cpu_usage=$(check_cpu_utilization)

  echo "CPU utilization: ${cpu_usage}"

  # Check if CPU utilization is less than 5% (you can adjust this threshold if needed)
  if [ "$(echo "$cpu_usage < 5.0" | bc -l)" -eq 1 ]; then
    # Increment the idle counter
    ((count+=1))

    echo "Count=${count}; waiting time=${wait}"

    if [ "$count" -gt "$wait" ]; then
      echo "CPU utilization is less than 5% for $wait minutes. Shutting down the VM."
      # Send a shutdown signal to the VM
      sudo shutdown -h now
      break
    fi
  else
    # Reset the idle counter
    ((count=0))
  fi
done
