import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib

# Load your pre-trained model
model = joblib.load("C:\\Users\\aneesh\\Desktop\\Internship\\ML1\\lr.pkl")

# Mapping for categorical features (adjust based on your mapping)
region_mapping = {'Thane West': 0, 'Mira Road East': 1, 'Dombivali': 2, 'Kandivali East': 3, 'Kharghar': 4}
type_mapping = {'Apartment': 0, 'Studio Apartment': 1, 'Villa': 2, 'Independent House': 3}
status_mapping = {'Ready to move': 0, 'Under construction': 1}

def map_categorical(feature, mapping):
    return mapping.get(feature.lower(), 0)  # Default to 0 if not found in mapping

def make_prediction(input_data):
    try:
        # Map categorical features
        input_data['type'] = map_categorical(input_data['type'], type_mapping)
        input_data['region'] = map_categorical(input_data['region'], region_mapping)
        input_data['status'] = map_categorical(input_data['status'], status_mapping)

        prediction = model.predict([list(input_data.values())])[0]
        return prediction
    except Exception as e:
        # Handle any exceptions that might occur during prediction
        raise

def format_prediction(prediction):
    # Convert to units (Cr or Lakhs) based on magnitude
    if abs(prediction) >= 100:
        formatted_value = f"{prediction / 100:.2f} Cr"
    elif abs(prediction) <=100 :
        formatted_value = f"{prediction:.2f} Lakhs"


    return formatted_value




def predict():
    try:
        # Get input data from the form
        input_data = {
            'bhk': float(entry_bhk.get()),
            'type': combo_type.get(),
            'area': float(entry_area.get()),
            'region': combo_region.get(),
            'status': combo_status.get()
        }

        # Make predictions using your model
        prediction = make_prediction(input_data)

        # Format the prediction
        formatted_prediction = format_prediction(prediction)

        # Display the formatted prediction in the label
        prediction_label.config(text=f"Your House should cost: {formatted_prediction}")

        # Clear input fields
        # entry_bhk.delete(0, tk.END)
        # entry_area.delete(0, tk.END)

    except ValueError as ve:
        # Handle ValueError (e.g., if the conversion to float fails)
        messagebox.showerror("Error", f"Invalid input: {str(ve)}")

    except Exception as e:
        # Handle other exceptions
        messagebox.showerror("Error", f"Something went wrong: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("500x400")  # Set a larger window size
root.configure(bg='#ADD8E6')  # Set background color

# Create and place labels and entry fields
label_bhk = ttk.Label(root, text="Number of Bedrooms (BHK):")
entry_bhk = ttk.Entry(root)
label_area = ttk.Label(root, text="Area (in square feet):")
entry_area = ttk.Entry(root)

# Create labels, entry fields, and dropdown menus for Type, Region, and Status
label_type = ttk.Label(root, text="Type of Property:")
label_region = ttk.Label(root, text="Region:")
label_status = ttk.Label(root, text="Status:")

entry_type = ttk.Entry(root)
entry_region = ttk.Entry(root)
entry_status = ttk.Entry(root)

combo_type = ttk.Combobox(root, values=list(type_mapping.keys()))
combo_region = ttk.Combobox(root, values=list(region_mapping.keys()))
combo_status = ttk.Combobox(root, values=list(status_mapping.keys()))

# Center the input fields
center_frame = ttk.Frame(root, style='TFrame', padding=(10, 5, 10, 5), relief="solid", borderwidth=1)
center_frame.grid(row=5, column=0, columnspan=2, pady=20)

label_bhk.grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
entry_bhk.grid(row=0, column=1, padx=5, pady=5)
label_area.grid(row=1, column=0, sticky=tk.E, padx=5, pady=5)
entry_area.grid(row=1, column=1, padx=5, pady=5)

label_type.grid(row=2, column=0, sticky=tk.E, padx=5, pady=5)
combo_type.grid(row=2, column=1, padx=5, pady=5)

label_region.grid(row=3, column=0, sticky=tk.E, padx=5, pady=5)
combo_region.grid(row=3, column=1, padx=5, pady=5)

label_status.grid(row=4, column=0, sticky=tk.E, padx=5, pady=5)
combo_status.grid(row=4, column=1, padx=5, pady=5)

# Create a style for the button
style = ttk.Style()
style.configure('TButton', font=('Arial', 12))
# Create and place the "Predict" button with larger font
predict_button = ttk.Button(root, text="Predict", command=predict, style='TButton')
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

# Create a label to display the prediction with larger font
prediction_label = ttk.Label(root, text="", font=('Arial', 14))
prediction_label.grid(row=7, column=0, columnspan=2, pady=10)

# Run the main loop
root.mainloop()
