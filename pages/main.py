import streamlit as st

def page1():
    st.title('Page 1')
    st.write('This is the first page of my app.')

def page2():
    st.title('Page 2')
    st.write('This is the second page of my app.')

# Define a dictionary that maps the user's selection in the sidebar to the corresponding page function
pages = {
    "Page 1": page1,
    "Page 2": page2
}

# Create a sidebar menu with radio buttons
selection = st.sidebar.selectbox("Go to", list(pages.keys()))

# Call the appropriate page function based on the user's selection
pages[selection]()

