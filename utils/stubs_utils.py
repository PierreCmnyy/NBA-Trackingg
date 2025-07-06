import os 
import pickle

def save_stub(stub_path, object) :
    """Saves the object to a stub file."""
    if not os.path.exists(os.path.dirname(stub_path)): # Check if the directory exists
        os.makedirs(os.path.dirname(stub_path)) # Create the directory if it doesn't exist
    if stub_path is not None: 
        with open(stub_path, 'wb') as f: # Open the file in binary write mode
            pickle.dump(object, f) # Save the object to the file (serialize it)

def read_stub(read_from_stub, stub_path):
    """Reads the object from a stub file."""
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f: # Open the file in binary read mode
            object = pickle.load(f)  # Load the object from the file (deserialize it)
            return object

    return None
