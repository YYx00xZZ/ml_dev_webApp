import streamlit as st
import re
# collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
numbers = st.text_input("PLease enter numbers")
st.write(collect_numbers(numbers))
st.write(np.array(collect_numbers(numbers)))