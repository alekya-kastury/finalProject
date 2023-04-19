import streamlit as st
from st_clickable_images import clickable_images

clicked = clickable_images(
    ["https://icons.iconarchive.com/icons/custom-icon-design/pretty-office-7/256/Save-icon.png"],
    div_style={"display": "flex", "justify-content": "left", "flex-wrap": "wrap"},
    img_style={"margin": "10px", "height": "250px"},
    key="clicked_images",
)

if clicked == 0:
    st.subheader("Saving Report..")
else:
    st.subheader(f"Selected Image No: #{clicked}")
