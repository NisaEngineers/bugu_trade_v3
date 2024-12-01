import os
# Function to write output to file and print it
def write_output_to_file(output, filename):
    # Ensure the "analysis" folder exists
    if not os.path.exists("analysis"):
        os.makedirs("analysis")
    
    filepath = os.path.join("analysis", filename)
    
    with open(filepath, 'w') as f:
        f.write(output)
    
    with open(filepath, 'r') as f:
        print(f.read())