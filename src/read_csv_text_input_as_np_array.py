import streamlit as st
import re
import numpy as np

# collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
numbers = st.text_input("PLease enter numbers")
st.write(collect_numbers(numbers))
st.write(np.array(collect_numbers(numbers)))
#4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820