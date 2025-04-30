## Author : Gabriel Gattaux
#!/bin/bash

# This will cause bash to stop executing the script if there's an error
set -e

# Define the parameters
irl_path="../data/raw/images_offline/images_100523_AVM_line_mapping/Expl4"
it_path="../data/raw/images_offline/images_100523_AVM_line_mapping/all"
param_values="10000,4,5,0.01,10000,4,5,0.01"
b=5  # constant parameter for osl
vis_values="32,3"
sav=0

script=src/train_test_mapping.py

# List of osl values to iterate over
osl_values_list=(2)

# Log the parameters in a CSV file
# Define the CSV file path
csv_file="../data/processed/mapping_parameters.csv"

# Create the CSV file with a header if it doesn't exist
if [ ! -f "$csv_file" ]; then
    echo "irl_path,it_path,param_values,osl_values,vis_values,unique_id" > "$csv_file"
fi

# Loop over each osl value
for osl in "${osl_values_list[@]}"; do
    osl_values="${osl},${b}"

    # Create the header line if the file doesn't exist
    if [ ! -f "$csv_file" ]; then
        echo "irl_path,it_path,param_values,osl_values,vis_values,unique_id" > "$csv_file"
    fi

    # Create a unique identifier for each set of parameters
    unique_id=$(echo -n "${irl_path},${it_path},${param_values},${osl_values},${vis_values}" | md5sum | cut -c 1-8)

    # Combine parameters to create a unique identifier (e.g., using a hash function)
    new_line="${irl_path},${it_path},${param_values},${osl_values},${vis_values},${unique_id}"

    # Create the output filename using the unique identifier
    output_file="../data/processed/mapping_${unique_id}"

    # Use sed to insert the new line after the header line
    sed -i "1 a $new_line" "$csv_file"

    # Run the Python script with the specified parameters
    python3 $script -irl $irl_path -it $it_path -o $output_file -p $param_values -osl $osl_values -vis $vis_values -sav $sav

    echo "Script execution completed for osl=${osl},${b}."
done

