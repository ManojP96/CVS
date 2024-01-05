import streamlit as st

# Initialize variables
num_rows = 3
num_columns = 3
variables = ["Variable 1", "Variable 2", "Variable 3"]
renamed_variables = {}
bucket_names = ["Bucket 1", "Bucket 2", "Bucket 3"]
Categorised_data = {variable: {'VB': 'Bucket 1', 'MB': None, 'BB': None} for variable in variables}
bucket_data_VB = {
    "Bucket 1": {'MB': 10, 'BB': 20},
    "Bucket 2": {'MB': 15, 'BB': 25},
    "Bucket 3": {'MB': 20, 'BB': 30},
}

# Create a Streamlit form
with st.form("variable_renaming_form"):
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx, col in enumerate(cols):
            variable_idx = row * num_columns + col_idx

            if variable_idx < len(variables):
                variable = variables[variable_idx]
                # Checkbox for renaming variables
                rename_checkbox = col.checkbox(f"Rename {variable}", key=f"{variable}_rename")

                # Use custom HTML and JavaScript to show/hide the text input when the checkbox is clicked
                col.write(variable)
                input_id = f"{variable}_input"
                col.write(f'<div id="{input_id}" style="display: none;">')
                col.write(f'<input type="text" id="{variable}_new_name" placeholder="New Name for {variable}">')
                col.write('</div>')
                col.write(f"""
                    <script>
                        var checkbox = document.getElementById("{variable}_rename");
                        var inputDiv = document.getElementById("{input_id}");
                        checkbox.addEventListener("change", function() {{
                            inputDiv.style.display = checkbox.checked ? "block" : "none";
                        }});
                    </script>
                """)

            # Selectbox for changing variable bucket
            bucket_key = f"{variable}_bucket"
            new_bucket_name = col.selectbox(f"Change Bucket Name for {variable}:", bucket_names, index=bucket_names.index(Categorised_data[variable]['VB']), key=bucket_key)
            Categorised_data[variable]['VB'] = new_bucket_name
            Categorised_data[variable]['MB'] = bucket_data_VB[new_bucket_name]['MB']
            Categorised_data[variable]['BB'] = bucket_data_VB[new_bucket_name]['BB']

    submitted_button = st.form_submit_button("Rename selected variables")

if submitted_button:
    st.write('Renamed variables:')
    st.write(renamed_variables)


















